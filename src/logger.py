import logging
import os

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/camera.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def get_logger():
    return logging