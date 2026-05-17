"""
Auris Cloud Server - Frames Router
Handles incoming motion crops from edge devices, runs heavy YOLOv8 inference,
DeepSort tracking, zone assignment, entry/exit counting, and stores analytical 
blobs in Cosmos DB.
"""

import io
import os
import time
import base64
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

import cv2
import numpy as np
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Try to import DB models (mocking import if db.py is not present locally yet)
try:
    from db import db, get_store_by_api_key
except ImportError:
    # Fallback for local testing if db.py is missing
    db = None
    async def get_store_by_api_key(api_key: str):
        return {"store_id": "sharma_karolbagh", "name": "Test Store"}

# Try loading heavy AI libraries
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None


logger = logging.getLogger("AurisCloud.Frames")
router = APIRouter()

# --- Module-Level AI Model Loading ---
MODEL = None
FIRE_MODEL = None
trackers: Dict[str, Any] = {}  # store_id_camera_id -> DeepSort instance

def get_tracker(store_id: str, camera_id: str):
    """Returns a unique DeepSort tracker per camera stream."""
    if not DeepSort:
        return None
    key = f"{store_id}_{camera_id}"
    if key not in trackers:
        trackers[key] = DeepSort(max_age=150)
    return trackers[key]

try:
    if YOLO:
        print("Loading heavy YOLOv8m model...", flush=True)
        MODEL = YOLO("yolov8m.pt")
        
        logger.info("Loading Fire Detection model...")
        # If fire.pt is missing, ultralytics might download a default or fail.
        # Wrapped in its own try/except just in case.
        try:
            FIRE_MODEL = YOLO("fire.pt")
        except Exception as e:
            logger.warning(f"Failed to load fire.pt: {e}")
            FIRE_MODEL = None
except Exception as e:
    logger.error(f"Error loading AI models: {e}")

async def send_fire_alert(store_id: str, camera_id: str, timestamp: str):
    """Background task to alert staff of detected fires."""
    # WhatsApp integration comes later
    logger.critical(f"🔥🔥 FIRE ALERT: Detected at {store_id} / {camera_id} at {timestamp} 🔥🔥")


# --- In-Memory State for Tracking & Counting ---
# Resets on server restart. Cosmos DB is the source of truth for historical data.
# Structure: store_id -> camera_id -> track_id -> value
def nested_dict():
    return defaultdict(lambda: defaultdict(dict))

def int_dict():
    return defaultdict(lambda: defaultdict(int))

track_positions = nested_dict()       # {store: {camera: {track_id: "zone_name"}}}
track_y_positions = nested_dict()     # {store: {camera: {track_id: float(y_centroid)}}}
count_in = int_dict()                 # {store: {camera: int}}
count_out = int_dict()                # {store: {camera: int}}
zone_entry_times = nested_dict()      # {store: {camera: {track_id: float(timestamp)}}}


# --- Pydantic Schemas ---
class CropPayload(BaseModel):
    bbox: List[float]
    jpeg_b64: str
    area: int

class FramePayload(BaseModel):
    store_id: str
    camera_id: str
    timestamp: str
    frame_id: int
    frame_resolution: List[int]
    calibration_mode: bool
    crops: List[CropPayload]
    full_frame_b64: Optional[str] = None


# --- Helper Functions ---
def run_inference_and_tracking(payload: FramePayload) -> Tuple[List[Dict[str, Any]], bool]:
    """
    CPU/GPU blocking function that runs YOLO and Fire models.
    Designed to be run in an async executor to prevent blocking the event loop.
    Returns: (List of detections for deepsort, bool indicating if fire detected)
    """
    if not MODEL:
        return [], False

    deepsort_detections = []
    fire_detected = False
    
    W_orig, H_orig = payload.frame_resolution
    
    for crop in payload.crops:
        try:
            # 1. Decode Crop
            img_data = base64.b64decode(crop.jpeg_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            crop_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if crop_img is None or crop_img.size == 0:
                continue
                
            crop_h, crop_w = crop_img.shape[:2]
            
            # Crop bounding box in original frame (normalized 0-1)
            cx_norm, cy_norm, cw_norm, ch_norm = crop.bbox
            
            # 2. Fire Detection
            if FIRE_MODEL:
                fire_results = FIRE_MODEL(crop_img, conf=0.5, verbose=False)
                for r in fire_results:
                    if len(r.boxes) > 0:
                        fire_detected = True
                        break
                        
            # 3. Person Detection
            results = MODEL(crop_img, conf=0.45, classes=[0], verbose=False)
            
            for r in results:
                for box in r.boxes:
                    # Bounding box local to the crop
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    
                    # Convert local crop pixels to crop-normalized (0-1)
                    nx1, ny1 = x1 / crop_w, y1 / crop_h
                    nx2, ny2 = x2 / crop_w, y2 / crop_h
                    
                    # Map to full-frame normalized coords (0-1)
                    # cx_norm, cy_norm is the top-left of the crop based on our Edge Worker design
                    # (In edge_worker.py, bbox was [px1/W, py1/H, w/W, h/H])
                    ff_nx1 = cx_norm + (nx1 * cw_norm)
                    ff_ny1 = cy_norm + (ny1 * ch_norm)
                    ff_nw = (nx2 - nx1) * cw_norm
                    ff_nh = (ny2 - ny1) * ch_norm
                    
                    # DeepSort requires [left, top, w, h] format in pixels for tracking
                    abs_x = max(0, min(int(ff_nx1 * W_orig), W_orig - 1))
                    abs_y = max(0, min(int(ff_ny1 * H_orig), H_orig - 1))
                    abs_w = max(1, int(ff_nw * W_orig))
                    abs_h = max(1, int(ff_nh * H_orig))
                    
                    # Prevent w/h from spilling out of bounds
                    if abs_x + abs_w > W_orig: abs_w = W_orig - abs_x
                    if abs_y + abs_h > H_orig: abs_h = H_orig - abs_y
                    
                    # Format: ([left, top, w, h], confidence, detection_class)
                    deepsort_detections.append(([abs_x, abs_y, abs_w, abs_h], conf, "person"))
                    
        except Exception as e:
            logger.warning(f"Failed to process a crop: {e}")
            continue
            
    if deepsort_detections:
        print(f"RAW YOLO: Detected {len(deepsort_detections)} people across {len(payload.crops)} crops (Before DeepSort tracking)", flush=True)
            
    return deepsort_detections, fire_detected


@router.post("/api/frames")
async def process_frame(request: Request, payload: FramePayload, background_tasks: BackgroundTasks):
    """
    Main endpoint for receiving and processing Edge Device frames.
    """
    try:
        # --- 1. AUTHENTICATION ---
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing X-API-Key")
            
        store = await get_store_by_api_key(api_key)
        if not store or store.get("store_id") != payload.store_id:
            raise HTTPException(status_code=401, detail="Invalid API Key or Store ID mismatch")
            
        s_id = payload.store_id
        c_id = payload.camera_id
        
        if not payload.crops and not payload.calibration_mode:
            # No crops = no motion = nothing to process
            # Still return 200 but skip inference entirely
            return {
                "status": "ok",
                "store_id": payload.store_id,
                "camera_id": payload.camera_id,
                "frame_id": payload.frame_id,
                "people_detected": 0,
                "fire_detected": False,
                "zone_events": [],
                "crossings": [],
                "calibration_frame_saved": False
            }
        
        # --- 2. YOLO & DEEPSORT INFERENCE (Thread Pool) ---
        loop = asyncio.get_event_loop()
        ds_detections, fire_detected = await loop.run_in_executor(None, run_inference_and_tracking, payload)
        
        if fire_detected:
            background_tasks.add_task(send_fire_alert, s_id, c_id, payload.timestamp)
        
        active_tracks = []
        tracker = get_tracker(s_id, c_id)
        if tracker and ds_detections:
            # Update tracks natively per camera using the exact original resolution
            W_orig, H_orig = payload.frame_resolution
            dummy_frame = np.zeros((H_orig, W_orig, 3), dtype=np.uint8)
            tracks = tracker.update_tracks(ds_detections, frame=dummy_frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb() # left, top, right, bottom (absolute pixels)
                
                # Convert back to normalized for zone math
                W_orig, H_orig = payload.frame_resolution
                nx1, ny1 = ltrb[0] / W_orig, ltrb[1] / H_orig
                nx2, ny2 = ltrb[2] / W_orig, ltrb[3] / H_orig
                
                # Centroid for zone calculation
                cx = (nx1 + nx2) / 2.0
                cy = (ny1 + ny2) / 2.0
                
                active_tracks.append({
                    "track_id": track_id,
                    "centroid": (cx, cy),
                    "bbox_normalised": [nx1, ny1, nx2, ny2],
                    "confidence": 0.9  # DeepSort doesn't retain conf natively in the track easily without custom patches
                })

        # --- 3. STORE CONFIGURATION & CALIBRATION ---
        zone_events = []
        crossings = []
        calib_saved = False
        
        # Fetch store zone configurations
        store_config = None
        if db is not None:
            store_config = await db.stores.find_one({"store_id": s_id})
            
        zone_config = store_config.get("zone_config", {}) if store_config else {}
        counting_line_y = store_config.get("counting_line_y", 0.5) if store_config else 0.5
        
        # --- 4. CALIBRATION MODE RETENTION ---
        if payload.calibration_mode and payload.full_frame_b64 and db is not None:
            # Save the full frame to Cosmos DB for auto_calibrator.py with a TTL (simulated by DB config)
            calib_doc = {
                "store_id": s_id,
                "camera_id": c_id,
                "timestamp": payload.timestamp,
                "frame_id": payload.frame_id,
                "full_frame_b64": payload.full_frame_b64,
                "created_at": datetime.now(timezone.utc)
            }
            await db.calibration_frames.insert_one(calib_doc)
            calib_saved = True

        # --- 5. LOGIC: ZONES & COUNTING ---
        current_frame_track_ids = set()
        final_detections = []
        
        now_ts = time.time()
        
        for t in active_tracks:
            tid = t["track_id"]
            cx, cy = t["centroid"]
            current_frame_track_ids.add(tid)
            
            # --- Counting Logic (Entrance/Exit Line Crossing) ---
            prev_y = track_y_positions[s_id][c_id].get(tid)
            if prev_y is not None:
                # 2-frame buffer zone safety check before counting
                if abs(cy - counting_line_y) > 0.08:
                    if prev_y < counting_line_y and cy >= counting_line_y:
                        count_in[s_id][c_id] += 1
                        crossings.append("entry")
                    elif prev_y > counting_line_y and cy <= counting_line_y:
                        count_out[s_id][c_id] += 1
                        crossings.append("exit")
            
            # Update memory
            track_y_positions[s_id][c_id][tid] = cy
            
            # --- Zone Assignment Logic ---
            current_zone = "unassigned"
            if zone_config:
                # Simple bounding box overlap check for zones: [x1, y1, x2, y2]
                for z_name, z_bbox in zone_config.items():
                    zx1, zy1, zx2, zy2 = z_bbox
                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        current_zone = z_name
                        break
                        
            prev_zone = track_positions[s_id][c_id].get(tid)
            
            if prev_zone != current_zone:
                # Track changed zones
                if prev_zone and prev_zone != "unassigned":
                    dwell = now_ts - zone_entry_times[s_id][c_id].get(tid, now_ts)
                    zone_events.append({
                        "track_id": tid,
                        "event": "exit",
                        "zone": prev_zone,
                        "dwell_seconds": round(dwell, 2)
                    })
                
                # Register new zone entry
                track_positions[s_id][c_id][tid] = current_zone
                zone_entry_times[s_id][c_id][tid] = now_ts
                
                if current_zone != "unassigned":
                    zone_events.append({
                        "track_id": tid,
                        "event": "entry",
                        "zone": current_zone
                    })
                    
            final_detections.append({
                "track_id": tid,
                "zone": current_zone,
                "confidence": t["confidence"],
                "bbox_normalised": t["bbox_normalised"]
            })

        if final_detections:
            logger.info(f"[{s_id}/{c_id}] frame {payload.frame_id}: "
                       f"{len(final_detections)} people detected, "
                       f"in={count_in[s_id][c_id]} out={count_out[s_id][c_id]}")

        # --- Clean up vanished tracks (Memory Management) ---
        known_tracks = list(track_positions[s_id][c_id].keys())
        for tid in known_tracks:
            if tid not in current_frame_track_ids:
                # Track disappeared
                prev_zone = track_positions[s_id][c_id].get(tid)
                if prev_zone and prev_zone != "unassigned":
                    dwell = now_ts - zone_entry_times[s_id][c_id].get(tid, now_ts)
                    zone_events.append({
                        "track_id": tid,
                        "event": "exit",
                        "zone": prev_zone,
                        "dwell_seconds": round(dwell, 2),
                        "reason": "disappeared"
                    })
                # Clear memory
                track_positions[s_id][c_id].pop(tid, None)
                track_y_positions[s_id][c_id].pop(tid, None)
                zone_entry_times[s_id][c_id].pop(tid, None)

        # --- 6. BUILD & SAVE BLOB TO COSMOS DB ---
        c_in = count_in[s_id][c_id]
        c_out = count_out[s_id][c_id]
        
        blob_doc = {
            "store_id": s_id,
            "camera_id": c_id,
            "timestamp": payload.timestamp,
            "frame_id": payload.frame_id,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "people_now": len(final_detections),
            "counts": {
                "in": c_in, 
                "out": c_out, 
                "current": max(0, c_in - c_out)
            },
            "zone_events": zone_events,
            "crossings": crossings,
            "fire_detected": fire_detected,
            "detections": final_detections
        }
        
        if db is not None:
            # Async insert into MongoDB/CosmosDB
            await db.blobs.insert_one(blob_doc)
            
        return {
            "status": "ok",
            "store_id": s_id,
            "camera_id": c_id,
            "frame_id": payload.frame_id,
            "people_detected": len(final_detections),
            "fire_detected": fire_detected,
            "zone_events": zone_events,
            "crossings": crossings,
            "calibration_frame_saved": calib_saved
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error processing frame {payload.frame_id}: {e}")
        # Never crash the endpoint, return whatever partial state we have
        return {
            "status": "error",
            "message": str(e),
            "store_id": payload.store_id,
            "camera_id": payload.camera_id,
            "frame_id": payload.frame_id
        }
