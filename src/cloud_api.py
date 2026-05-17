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

class BlobPayload(BaseModel):
    camera_id: str
    timestamp: str
    blob_image_b64: str
    bbox: list
    frame_resolution: list

class BlobBatch(BaseModel):
    blobs: list[BlobPayload]

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

@app.post("/api/blobs")
async def receive_blob(payload: BlobBatch, background_tasks: BackgroundTasks):
    try:
        detected_items = []
        for blob in payload.blobs:
            # 1. Decode the Base64 Blob back into an OpenCV Image
            img_data = base64.b64decode(blob.blob_image_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            # 2. Save Blob temporarily (Privacy Policy: Deleted after 30 mins)
            safe_time = blob.timestamp.replace(":", "-").replace(".", "-")
            filepath = os.path.join(BLOB_STORAGE_DIR, f"{blob.camera_id}_{safe_time}.jpg")
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
            if blob.camera_id not in FACTORY_BLOB_PATHS:
                FACTORY_BLOB_PATHS[blob.camera_id] = []
            
            # We store the center of the bounding box for tracking
            cx = (blob.bbox[0] + blob.bbox[2]) / 2
            cy = (blob.bbox[1] + blob.bbox[3]) / 2
            FACTORY_BLOB_PATHS[blob.camera_id].append([cx, cy])

            logger.info(f"[Cloud Catcher] Caught blob from {blob.camera_id}. YOLO Detected: [{detection_result.upper()}]")
            detected_items.append(detection_result)

        return {"status": "success", "detected": detected_items}

    except Exception as e:
        logger.error(f"Error processing blob: {e}")
        return {"status": "error"}

@app.get("/api/status")
def status():
    return {"status": "online", "model_loaded": MODEL is not None}
