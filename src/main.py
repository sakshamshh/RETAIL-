import time
import logging
from logger import get_logger
from blob_emitter import BlobEmitter
from camera_worker import CameraWorker
from updater import OTAUpdater

def main():
    logger = get_logger()
    logger.info("Initializing Auris Edge Node...")

    # 1. Start OTA Updater (Checks GitHub every 60 mins for new code)
    updater = OTAUpdater(check_interval_seconds=3600, logger=logger)
    updater.start()

    # 2. Start Blob Emitter (Queue that POSTs images to Azure Cloud)
    emitter = BlobEmitter(logger=logger)
    
    # 3. Start Camera Worker (OpenCV Background Subtraction -> Blob Cropping)
    # Using the provided test factory video for the simulation
    video_path = r"C:\Users\SAKSHAM\Downloads\testvid1.mp4"
    camera = CameraWorker(name="edge_node_1", url=video_path, target_fps=15, logger=logger, blob_emitter=emitter)
    
    try:
        # This will block and run infinitely
        camera.start()
    except KeyboardInterrupt:
        logger.info("Shutting down Edge Node...")
        camera.stop()
        updater.stop()

if __name__ == "__main__":
    main()