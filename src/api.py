import asyncio
import threading
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.websocket_manager import broadcast, connect, disconnect
from src.config import CAMERAS
from src.camera_worker import CameraWorker
from src.logger import get_logger

app = FastAPI()
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
logger = get_logger()
loop = None

def notify(msg):
    logger.warning(msg)
    if loop:
        asyncio.run_coroutine_threadsafe(
            broadcast({"type": "alert", "message": msg}),
            loop
        )

def notify_frame(cam_name, b64_frame):
    if loop:
        asyncio.run_coroutine_threadsafe(
            broadcast({"type": "frame", "camera": cam_name, "data": b64_frame}),
            loop
        )

def notify_stats(stats):
    if loop:
        asyncio.run_coroutine_threadsafe(
            broadcast(stats),
            loop
        )

@app.on_event("startup")
async def startup():
    global loop
    loop = asyncio.get_event_loop()
    for name, cfg in CAMERAS.items():
        worker = CameraWorker(name, cfg["url"], cfg["fps"], logger, notify, notify_frame, notify_stats)
        worker.daemon = True
        worker.start()

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        disconnect(websocket)
