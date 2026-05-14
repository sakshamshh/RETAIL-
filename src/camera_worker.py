import cv2
import time
import threading
import base64
from src.detector import detect_people, draw_boxes
from src.analytics import analytics

class CameraWorker(threading.Thread):
    def __init__(self, name, url, target_fps, logger, notify, notify_frame, notify_stats):
        super().__init__()
        self.name = name
        self.url = url
        self.target_fps = target_fps
        self.frame_interval = 1 / target_fps
        self.running = True
        self.cap = None
        self.logger = logger
        self.notify = notify
        self.notify_frame = notify_frame
        self.notify_stats = notify_stats

    def connect(self):
        while True:
            if isinstance(self.url, int):
                self.cap = cv2.VideoCapture(self.url)
            else:
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if self.cap.isOpened():
                self.logger.info(f"{self.name} connected")
                return
            self.notify(f"{self.name} connection failed, retrying...")
            time.sleep(2)

    def run(self):
        self.connect()
        last_frame_time = 0

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            now = time.time()

            if now - last_frame_time >= self.frame_interval:
                last_frame_time = now
                boxes = detect_people(frame)
                frame = draw_boxes(frame, boxes)
                count = len(boxes)
                stats = analytics.update(self.name, count)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buffer).decode('utf-8')
                self.notify_frame(self.name, b64)
                self.notify_stats(stats)
                for alert in stats["alerts"]:
                    self.notify(alert)
                self.logger.info(f"{self.name}: {count} people | {stats['period']} | peak: {stats['peak_hour']}")
            else:
                time.sleep(self.frame_interval - (now - last_frame_time))
