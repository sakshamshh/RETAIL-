import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from src.websocket_manager import broadcast, connect, disconnect
from src.config import CAMERAS
from src.camera_worker import CameraWorker
from src.logger import get_logger
from src.database import init_db, get_hourly_traffic, get_daily_summary, get_today_stats

app = FastAPI()
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
logger = get_logger()
loop = None

def notify(msg):
    logger.warning(msg)
    if loop:
        asyncio.run_coroutine_threadsafe(broadcast({"type": "alert", "message": msg}), loop)

def notify_frame(cam_name, b64_frame):
    if loop:
        asyncio.run_coroutine_threadsafe(broadcast({"type": "frame", "camera": cam_name, "data": b64_frame}), loop)

def notify_stats(stats):
    if loop:
        asyncio.run_coroutine_threadsafe(broadcast(stats), loop)

@app.on_event("startup")
async def startup():
    global loop
    loop = asyncio.get_event_loop()
    init_db()
    for name, cfg in CAMERAS.items():
        worker = CameraWorker(name, cfg["url"], cfg["fps"], logger, notify, notify_frame, notify_stats)
        worker.daemon = True
        worker.start()

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.get("/api/today")
async def today():
    return JSONResponse(get_today_stats())

@app.get("/api/hourly")
async def hourly():
    data = get_hourly_traffic()
    return JSONResponse([{"hour": r[0], "avg": round(r[1], 1)} for r in data])

@app.get("/api/daily")
async def daily():
    data = get_daily_summary()
    return JSONResponse([{"day": r[0], "avg": round(r[1], 1), "entries": r[2], "exits": r[3]} for r in data])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        disconnect(websocket)
