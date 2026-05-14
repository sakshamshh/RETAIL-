import cv2
import time
import threading
import base64
from src.detector import detect_people
from src.analytics import analytics
from src.tracker import update_tracks
from src.entry_exit import get_counter
from src.database import save_traffic, save_alert

class CameraWorker(threading.Thread):
    def __init__(self, name, url, target_fps, logger, notify, notify_frame, notify_stats):
        super().__init__()
        self.name = name
        self.url = url
        self.target_fps = target_fps
        self.frame_interval = 1 / target_fps
        self.stats_interval = 2
        self.db_interval = 30
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
        last_stats_time = 0
        last_db_time = 0
        counter = get_counter(self.name)

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            now = time.time()

            if now - last_frame_time >= self.frame_interval:
                last_frame_time = now
                boxes = detect_people(frame)
                tracks = update_tracks(self.name, boxes, frame)
                entry_stats = counter.update(tracks, frame.shape[0])

                for (tid, x1, y1, x2, y2) in tracks:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                line_y = int(frame.shape[0] * 0.5)
                cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
                cv2.putText(frame, f"IN:{entry_stats['entries']} OUT:{entry_stats['exits']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                count = len(tracks)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                b64 = base64.b64encode(buffer).decode('utf-8')
                self.notify_frame(self.name, b64)

                if now - last_stats_time >= self.stats_interval:
                    last_stats_time = now
                    stats = analytics.update(self.name, count)
                    stats["entries"] = entry_stats["entries"]
                    stats["exits"] = entry_stats["exits"]
                    stats["net"] = entry_stats["net"]
                    self.notify_stats(stats)
                    for alert in stats["alerts"]:
                        self.notify(alert)
                        save_alert(self.name, alert)

                if now - last_db_time >= self.db_interval:
                    last_db_time = now
                    save_traffic(self.name, count, entry_stats["entries"], entry_stats["exits"], entry_stats["net"])
                    self.logger.info(f"{self.name}: saved to DB | people={count} IN={entry_stats['entries']}")
            else:
                time.sleep(self.frame_interval - (now - last_frame_time))
