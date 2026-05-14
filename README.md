# RetailIQ - AI Retail Intelligence Platform

Real time AI powered surveillance and analytics for retail stores.

## Features
- Live camera feed streaming (RTSP or video file)
- People detection using YOLOv8
- Entry/exit counting
- Peak hour detection
- Weekday vs weekend analytics
- Store open/closed monitoring
- Capacity and crowding alerts
- Multi camera support
- Real time dashboard

## Requirements
- Python 3.10+
- Webcam or RTSP camera or video file

## Installation
```bash
pip install -r requirements.txt
```

## Configuration
Edit `.env` to set:
- MAX_CAPACITY — maximum store capacity
- STORE_OPEN / STORE_CLOSE — store hours
- FPS — frames per second per camera

Edit `src/config.py` to add your camera URLs.

## Run
```bash
uvicorn src.api:app --reload
```

Open browser at http://127.0.0.1:8000

## Camera Setup
Edit `src/config.py`:
```python
CAMERAS = {
    "cam1": {"url": "rtsp://your-camera-ip/stream", "fps": 5},
}
```

## Version
v1.0.0
