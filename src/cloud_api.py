import os
import cv2
import time
import base64
import asyncio
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging

try:
    from ultralytics import YOLO
    MODEL = YOLO("yolov8n.pt")
except ImportError:
    MODEL = None

app = FastAPI(title="Auris Cloud Brain API")
logger = logging.getLogger("CloudAPI")
logging.basicConfig(level=logging.INFO)

# In-memory database for simulation tracking
FACTORY_BLOB_PATHS = {}

# The directory to store blobs for 30 minutes
BLOB_STORAGE_DIR = os.path.join(os.path.dirname(__file__), "temp_blobs")
os.makedirs(BLOB_STORAGE_DIR, exist_ok=True)

class CropPayload(BaseModel):
    bbox: list
    jpeg_b64: str
    area: int

class FramePayload(BaseModel):
    store_id: str
    camera_id: str
    timestamp: str
    frame_id: int
    frame_resolution: list
    calibration_mode: bool
    crops: list[CropPayload]
    full_frame_b64: Optional[str] = None

async def delete_old_blobs():
    """Background task that runs continuously to delete blobs older than 30 mins."""
    while True:
        try:
            now = time.time()
            for filename in os.listdir(BLOB_STORAGE_DIR):
                filepath = os.path.join(BLOB_STORAGE_DIR, filename)
                if os.path.isfile(filepath):
                    # Strict 30-minute privacy retention policy
                    if os.stat(filepath).st_mtime < now - (30 * 60):
                        os.remove(filepath)
                        logger.info(f"[Privacy Engine] Shredded expired blob: {filename}")
        except Exception as e:
            logger.error(f"Error in privacy cleanup: {e}")
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    # Start the 30-min privacy retention sweeper
    logger.info("[Privacy Engine] 30-Minute Blob Shredder initialized.")
    asyncio.create_task(delete_old_blobs())

@app.post("/api/frames")
async def receive_frame(payload: FramePayload, background_tasks: BackgroundTasks):
    try:
        detected_items = []
        safe_time = payload.timestamp.replace(":", "-").replace(".", "-")
        
        # 1. Handle Calibration Full Frame
        if payload.calibration_mode and payload.full_frame_b64:
            img_data = base64.b64decode(payload.full_frame_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            full_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if full_img is not None:
                calib_path = os.path.join(BLOB_STORAGE_DIR, f"calib_{payload.camera_id}_{safe_time}.jpg")
                cv2.imwrite(calib_path, full_img)
                logger.info(f"[Calibration] Saved full frame for {payload.camera_id}")

        # 2. Process Crops
        for i, crop in enumerate(payload.crops):
            img_data = base64.b64decode(crop.jpeg_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            # Save Crop temporarily (Privacy Policy: Deleted after 30 mins)
            filepath = os.path.join(BLOB_STORAGE_DIR, f"crop_{payload.camera_id}_{safe_time}_{i}.jpg")
            cv2.imwrite(filepath, img)

            # 3. Heavy-Duty YOLO Inference
            detection_result = "Unknown Motion"
            if MODEL:
                results = MODEL(img, verbose=False)
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = MODEL.names[cls]
                        
                        if conf > 0.60:
                            detection_result = name
                            break

            # 4. Math Aggregation for Zero-Click Calibration
            if payload.camera_id not in FACTORY_BLOB_PATHS:
                FACTORY_BLOB_PATHS[payload.camera_id] = []
            
            # We store the center of the bounding box for tracking
            # Note: bbox is normalized [x, y, w, h]
            x, y, w, h = crop.bbox
            cx = x + (w / 2)
            cy = y + (h / 2)
            FACTORY_BLOB_PATHS[payload.camera_id].append([cx, cy])

            logger.info(f"[Cloud Catcher] Processed crop from {payload.camera_id}. YOLO Detected: [{detection_result.upper()}]")
            detected_items.append(detection_result)

        return {"status": "success", "detected": detected_items, "calib_frame_saved": bool(payload.full_frame_b64)}

    except Exception as e:
        logger.error(f"Error processing blob: {e}")
        return {"status": "error"}

@app.get("/api/status")
def status():
    return {"status": "online", "model_loaded": MODEL is not None}
