import cv2
import time
import threading

class CameraWorker(threading.Thread):
    def __init__(self, name, url, target_fps, logger, notify):
        super().__init__()
        self.name = name
        self.url = url
        self.target_fps = target_fps
        self.frame_interval = 1 / target_fps
        self.running = True
        self.cap = None
        self.logger = logger
        self.notify = notify

    def connect(self):
        while True:
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

            if self.cap.isOpened():
                self.logger.info(f"{self.name} connected")
                return

            self.notify(f"{self.name} connection failed, retrying...")
            time.sleep(2)

    def run(self):
        self.connect()

        last_frame_time = 0
        fail_count = 0

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                fail_count += 1
                self.logger.error(f"{self.name} frame read failed")

                if fail_count >= 5:
                    self.notify(f"{self.name} stream lost, reconnecting...")
                    self.cap.release()
                    self.connect()
                    fail_count = 0

                time.sleep(0.5)
                continue

            fail_count = 0
            now = time.time()

            # FPS sampling
            if now - last_frame_time >= self.frame_interval:
                last_frame_time = now
                self.logger.info(f"{self.name} frame processed")

                #cv2.imshow(self.name, frame)

            if cv2.waitKey(1) == 27:
                self.running = False