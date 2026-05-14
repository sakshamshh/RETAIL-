from camera_worker import CameraWorker
from config import CAMERAS
from src.logger import get_logger
#import asyncio
from src.websocket_manager import broadcast
logger = get_logger()
import asyncio

loop = asyncio.get_event_loop()
def notify(msg):

    logger.warning(msg)

    asyncio.run_coroutine_threadsafe(
        broadcast({
            "message": msg
        }),
        loop
    )

workers = []

for name, cfg in CAMERAS.items():
    worker = CameraWorker(
        name,
        cfg["url"],
        cfg["fps"],
        logger,
        notify
    )
    worker.start()
    workers.append(worker)

for w in workers:
    w.join()