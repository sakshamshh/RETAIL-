import cv2
import time
import base64
import logging
from datetime import datetime, timezone

class CameraWorker:
    def __init__(self, name, url, target_fps, logger, blob_emitter, **kwargs):
        self.name = name
        # Support "0" for local webcam testing
        self.url = int(url) if str(url).isdigit() else url
        self.target_fps = target_fps
        self.logger = logger or logging.getLogger(__name__)
        self.blob_emitter = blob_emitter
        self.running = False
        
        # MOG2 Background Subtractor - ultra lightweight
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.min_blob_area = 500  # Minimum pixel area to consider a "blob"

    def start(self):
        self.running = True
        self.logger.info(f"[{self.name}] Starting pure Blob Emitter on source: {self.url}")
        cap = cv2.VideoCapture(self.url)
        
        if not cap.isOpened():
            self.logger.error(f"[{self.name}] Failed to open camera. Make sure webcam is not used by another app.")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.logger.warning(f"[{self.name}] Frame drop. Retrying...")
                time.sleep(1)
                cap = cv2.VideoCapture(self.url)
                continue

            # 1. Background Subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # 2. Noise removal (morphology)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # 3. Find moving contours (blobs)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            blobs_found = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > self.min_blob_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Pad the crop slightly for YOLO context
                    pad = 15
                    y1 = max(0, y - pad)
                    y2 = min(frame.shape[0], y + h + pad)
                    x1 = max(0, x - pad)
                    x2 = min(frame.shape[1], x + w + pad)
                    
                    blob_crop = frame[y1:y2, x1:x2]
                    
                    # Encode cropped blob to base64 JPEG
                    _, buffer = cv2.imencode('.jpg', blob_crop)
                    b64_img = base64.b64encode(buffer).decode('utf-8')
                    
                    # Package payload for Azure Cloud Server
                    payload = {
                        "camera_id": self.name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "blob_image_b64": b64_img,
                        "bbox": [x1, y1, x2, y2],
                        "frame_resolution": [frame.shape[1], frame.shape[0]]
                    }
                    
                    # Queue for upload
                    if self.blob_emitter:
                        self.blob_emitter.enqueue(payload)
                        blobs_found += 1
            
            if blobs_found > 0:
                self.logger.debug(f"[{self.name}] Extracted and queued {blobs_found} motion blobs")
            
            # Respect target FPS to save CPU
            if self.target_fps > 0:
                time.sleep(1.0 / self.target_fps)

        cap.release()

    def stop(self):
        self.running = False
