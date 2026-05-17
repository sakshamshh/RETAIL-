"""
Auris Edge Worker
Production-grade edge device application for Skym Labs.
Performs motion detection (MOG2), crop extraction, and cloud synchronization
for the Auris AI computer vision platform.

Runs 24/7 on Intel N100 mini PCs.
"""

import os
import time
import json
import base64
import sqlite3
import threading
import logging
import queue
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Any, Optional

import cv2
import numpy as np
import requests
from dotenv import load_dotenv, set_key

# --- Configuration Paths ---
# Defaults to /opt/auris/ but handles Windows local dev gracefully
BASE_DIR = "/opt/auris" if os.name != "nt" else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
DB_PATH = os.path.join(BASE_DIR, "data", "frame_buffer.db")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "edge.log")

# Ensure directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load environment variables
load_dotenv(ENV_PATH)

CLOUD_ENDPOINT = os.getenv("CLOUD_ENDPOINT", "https://auris.skymlabs.com/api/frames")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "")
STORE_ID = os.getenv("STORE_ID", "default_store")
CALIBRATION_START_STR = os.getenv("CALIBRATION_START", datetime.now(timezone.utc).isoformat())

# Set up logging
logger = logging.getLogger("AurisEdge")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s')

# Console Handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# File Handler (Daily rotation, keep 7 days)
fh = TimedRotatingFileHandler(LOG_PATH, when="midnight", interval=1, backupCount=7)
fh.setFormatter(formatter)
logger.addHandler(fh)

_calibration_cache = None
_calibration_checked = None

def is_calibration_mode() -> bool:
    """
    Checks if the device should be in CALIBRATION mode.
    Calibration mode lasts for exactly 48 hours from CALIBRATION_START.
    """
    global _calibration_cache, _calibration_checked
    now = datetime.now(timezone.utc)
    
    # Re-check only every 60 seconds
    if _calibration_checked and (now - _calibration_checked).total_seconds() < 60:
        return _calibration_cache
        
    _calibration_checked = now
    
    mode = os.getenv("CALIBRATION_MODE", "true").lower() == "true"
    if not mode:
        _calibration_cache = False
        return False
        
    try:
        # Try parsing ISO 8601
        try:
            start_time = datetime.fromisoformat(CALIBRATION_START_STR.replace('Z', '+00:00'))
        except ValueError:
            # Fallback for simpler formats if needed
            start_time = datetime.now(timezone.utc)
            
        if now - start_time > timedelta(hours=48):
            logger.info("48-hour calibration period ended. Switching to NORMAL mode.")
            os.environ["CALIBRATION_MODE"] = "false"
            # Update the .env file permanently
            try:
                set_key(ENV_PATH, "CALIBRATION_MODE", "false")
            except Exception as e:
                logger.error(f"Failed to update .env: {e}")
            _calibration_cache = False
            return False
            
        _calibration_cache = True
        return True
    except Exception as e:
        logger.error(f"Error parsing CALIBRATION_START: {e}. Defaulting to calibration mode.")
        _calibration_cache = True
        return True


class FrameUploader:
    """
    Handles buffering and uploading of blobs to the Cloud API.
    Provides offline SQLite buffering and automatic retry.
    """
    def __init__(self, endpoint: str, api_key: str, store_id: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.store_id = store_id
        self.queue = queue.Queue(maxsize=300)
        self._upload_semaphore = threading.Semaphore(2)
        self._buffer_size = 0
        
        self._init_db()
        
        # Start background threads
        threading.Thread(target=self._upload_loop, name="Uploader-Main", daemon=True).start()
        threading.Thread(target=self._retry_loop, name="Uploader-Retry", daemon=True).start()

    def _init_db(self):
        """Initializes the SQLite buffer database."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS frame_buffer (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        payload TEXT NOT NULL, 
                        created TEXT NOT NULL
                    )
                """)
                count = conn.execute("SELECT COUNT(*) FROM frame_buffer").fetchone()[0]
                self._buffer_size = count
        except Exception as e:
            logger.error(f"Failed to initialize SQLite buffer: {e}")

    def get_buffer_size(self):
        return self._buffer_size

    def enqueue(self, payload: Dict[str, Any]):
        """Adds a payload to the upload queue."""
        try:
            self.queue.put_nowait(payload)
        except __import__('queue').Full:
            logger.warning("Upload queue full, saving directly to buffer")
            self._buffer(payload)

    def _post(self, payload: Dict[str, Any]) -> bool:
        """Attempts to POST the payload to the cloud API."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            if payload.get("calibration_mode", False):
                headers["X-Calibration"] = "true"
                
            with self._upload_semaphore:
                resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=10)
            return resp.status_code in (200, 201, 202)
        except requests.exceptions.RequestException as e:
            logger.debug(f"HTTP Post failed: {e}")
            return False

    def _buffer(self, payload: Dict[str, Any]):
        """Saves a failed payload to the SQLite buffer. Drops oldest if full (>5000)."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # Enforce max buffer size of 5000
                if self._buffer_size >= 5000:
                    logger.warning("Buffer full! Dropping oldest frame.")
                    conn.execute("DELETE FROM frame_buffer WHERE id = (SELECT MIN(id) FROM frame_buffer)")
                    self._buffer_size -= 1
                
                conn.execute(
                    "INSERT INTO frame_buffer (payload, created) VALUES (?, ?)", 
                    (json.dumps(payload), datetime.now(timezone.utc).isoformat())
                )
                self._buffer_size += 1
        except Exception as e:
            logger.error(f"Failed to buffer payload: {e}")

    def _upload_loop(self):
        """Continuously processes the live queue."""
        logger.info("Live upload loop started.")
        while True:
            try:
                payload = self.queue.get()
                if not self._post(payload):
                    self._buffer(payload)
            except Exception as e:
                logger.error(f"Error in upload loop: {e}")
                time.sleep(1)

    def _retry_loop(self):
        """Continuously retries buffered payloads in batches."""
        logger.info("Buffer retry loop started.")
        while True:
            time.sleep(15) # Retry every 15 seconds
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    rows = conn.execute(
                        "SELECT id, payload FROM frame_buffer ORDER BY id ASC LIMIT 10"
                    ).fetchall()
                
                if not rows:
                    continue
                    
                logger.info(f"Retrying {len(rows)} buffered frames...")
                success_ids = []
                
                for row_id, payload_str in rows:
                    payload = json.loads(payload_str)
                    if self._post(payload):
                        success_ids.append(row_id)
                
                if success_ids:
                    with sqlite3.connect(DB_PATH) as conn:
                        placeholders = ",".join("?" for _ in success_ids)
                        conn.execute(f"DELETE FROM frame_buffer WHERE id IN ({placeholders})", success_ids)
                    self._buffer_size -= len(success_ids)
                    logger.info(f"Successfully recovered {len(success_ids)} frames from buffer.")
                    
            except Exception as e:
                logger.error(f"Error in retry loop: {e}")


class CameraWorker(threading.Thread):
    """
    Handles RTSP connection, MOG2 background subtraction, and crop extraction for a single camera.
    """
    def __init__(self, name: str, url: Any, target_fps: int, store_id: str, uploader: FrameUploader):
        super().__init__(name=name)
        self.url = url
        self.target_fps = target_fps
        self.store_id = store_id
        self.uploader = uploader
        
        self.cap = None
        self.running = False
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.frame_count = 0
        self.consecutive_failures = 0

    def get_effective_fps(self) -> float:
        queue_size = self.uploader.queue.qsize()
        max_size = self.uploader.queue.maxsize
        ratio = queue_size / max_size
        if ratio > 0.8:
            return 0.5
        elif ratio > 0.5:
            return 1.0
        elif ratio > 0.2:
            return min(self.target_fps, 2.0)
        else:
            return self.target_fps

    def connect(self):
        """Establishes connection to the RTSP stream."""
        if self.cap:
            self.cap.release()
        logger.info(f"Connecting to stream: {self.url}")
        
        # Use FFMPEG backend explicitly for robust RTSP handling
        if isinstance(self.url, str) and self.url.startswith("rtsp"):
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.url)
            
        if not self.cap.isOpened():
            logger.error(f"Failed to open stream {self.url}")
            return False
        return True

    def merge_overlapping_crops(self, rects: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merges overlapping bounding boxes using cv2.groupRectangles."""
        if not rects:
            return []
        
        # groupRectangles requires a list of rects and a weights array.
        # We duplicate rects to force grouping even if it's a single isolated rect.
        rects_list = list(rects) + list(rects)
        merged, _ = cv2.groupRectangles(rects_list, 1, 0.2)
        
        if merged is None or len(merged) == 0:
            return []
            
        return [tuple(r) for r in merged]

    def extract_crops(self, frame: np.ndarray, mask: np.ndarray) -> List[Dict[str, Any]]:
        """Finds motion contours, extracts padded crops, and returns payload dictionaries."""
        H, W = frame.shape[:2]
        
        # Find contours in the motion mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_rects = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 800:
                continue # Ignore dust, insects, tiny movements
            
            x, y, w, h = cv2.boundingRect(c)
            raw_rects.append((x, y, w, h))
            
        # Merge overlapping rectangles (e.g. people walking close to each other)
        merged_rects = self.merge_overlapping_crops(raw_rects)
        
        crops_data = []
        for (x, y, w, h) in merged_rects:
            x, y, w, h = int(x), int(y), int(w), int(h)
            area = w * h
            
            # Add 30px padding
            px1 = max(0, x - 30)
            py1 = max(0, y - 30)
            px2 = min(W, x + w + 30)
            py2 = min(H, y + h + 30)
            
            crop_img = frame[py1:py2, px1:px2]
            
            # Resize crop if larger than 320x320
            ch, cw = crop_img.shape[:2]
            if ch > 320 or cw > 320:
                scale = 320 / max(ch, cw)
                crop_img = cv2.resize(crop_img, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
                
            # JPEG Encode (quality=60)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            success, encoded_img = cv2.imencode('.jpg', crop_img, encode_param)
            
            if success:
                b64_str = base64.b64encode(encoded_img).decode('utf-8')
                
                # Normalise bbox coords 0-1
                norm_bbox = [float(px1/W), float(py1/H), float((px2-px1)/W), float((py2-py1)/H)]
                
                crops_data.append({
                    "bbox": norm_bbox,
                    "jpeg_b64": b64_str,
                    "area": int(area)
                })
                
        # Maximum 20 crops (take largest by area)
        crops_data.sort(key=lambda c: c["area"], reverse=True)
        
        # Filter out insignificant motion (background noise, dust, lighting changes)
        crops_data = [c for c in crops_data if c["area"] > 3000]
        
        # Only send if at least one substantial crop exists
        if not crops_data:
            return []
            
        return crops_data[:20]

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def run(self):
        self.running = True
        logger.info(f"Started camera worker for {self.name}")
        
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    if not self.connect():
                        time.sleep(2)
                        continue

                calibration_mode = is_calibration_mode()
                
                # Adjust FPS based on mode
                effective_fps = 2 if calibration_mode else self.get_effective_fps()
                frame_delay = 1.0 / effective_fps
                
                start_time = time.time()
                ret, frame = self.cap.read()
                
                if not ret:
                    self.consecutive_failures += 1
                    logger.warning(f"Failed to read frame from {self.name} ({self.consecutive_failures}/5)")
                    if self.consecutive_failures >= 5:
                        logger.error(f"Reconnecting stream {self.name} due to 5 failures")
                        self.connect()
                        self.consecutive_failures = 0
                    time.sleep(1)
                    continue
                    
                self.consecutive_failures = 0
                self.frame_count += 1
                
                if self.frame_count % 50 == 0:
                    # Log stats every 50 frames using the fast cached counter
                    buf_size = self.uploader.get_buffer_size()
                    logger.info(f"[{self.name}] Processed {self.frame_count} frames. Buffer size: {buf_size}")

                H, W = frame.shape[:2]
                
                # Apply Background Subtraction
                mask = self.bg_subtractor.apply(frame)
                
                # Extract crops
                crops = self.extract_crops(frame, mask)
                
                # Only construct and send payload if there is motion OR if we need to send full frames in calibration
                full_frame_b64 = None
                # Only send full frame every 10th frame in calibration mode to save bandwidth (~50KB/frame)
                if calibration_mode and self.frame_count % 10 == 0:
                    ff_resized = cv2.resize(frame, (640, 480))
                    _, ff_enc = cv2.imencode('.jpg', ff_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    full_frame_b64 = base64.b64encode(ff_enc).decode('utf-8')

                if crops or full_frame_b64:
                    payload = {
                        "store_id": self.store_id,
                        "camera_id": self.name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "frame_id": self.frame_count,
                        "frame_resolution": [W, H],
                        "calibration_mode": calibration_mode,
                        "crops": crops,
                        "full_frame_b64": full_frame_b64
                    }
                    self.uploader.enqueue(payload)
                
                # Control frame rate by sleeping remainder of the target delay
                elapsed = time.time() - start_time
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Unexpected error in camera worker {self.name}: {e}")
                time.sleep(2)


def main():
    logger.info("Initializing Auris Edge Service...")
    
    # Initialize the Uploader (handles HTTP and SQLite buffering)
    uploader = FrameUploader(endpoint=CLOUD_ENDPOINT, api_key=CLOUD_API_KEY, store_id=STORE_ID)
    
    # Load cameras from config (fallback to webcam if missing)
    try:
        import sys
        sys.path.append(os.path.join(BASE_DIR, "src"))
        from config import CAMERAS
    except ImportError:
        logger.warning("Could not load /opt/auris/src/config.py. Defaulting to video simulation.")
        CAMERAS = {
            "cam1": {"url": "videos/vid1.mp4", "fps": 2},
        }
        
    workers = []
    
    for cam_name, cam_cfg in CAMERAS.items():
        url = cam_cfg.get("url", 0)
        fps = cam_cfg.get("fps", 5)
        
        worker = CameraWorker(
            name=cam_name, 
            url=url, 
            target_fps=fps, 
            store_id=STORE_ID, 
            uploader=uploader
        )
        workers.append(worker)
        worker.start()
        
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Edge Service...")
        for w in workers:
            w.stop()
        for w in workers:
            w.join()

if __name__ == "__main__":
    main()
