import cv2, time, threading, collections
from datetime import datetime, timezone

CONFIDENCE_THRESHOLD = 0.45
ENTRY_ZONE_BUFFER    = 2
SMOOTHING_WINDOW     = 3
STAFF_IGNORE_MINUTES = 30
DEEPSORT_MAX_AGE     = 150

class ZoneTracker:
    def __init__(self):
        self._current_zone  = {}
        self._zone_entry_ts = {}
        self._events        = []

    def update(self, track_id, zone, now):
        prev_zone = self._current_zone.get(track_id)
        if prev_zone is None:
            self._current_zone[track_id]  = zone
            self._zone_entry_ts[track_id] = now
            self._events.append({"track_id": track_id, "zone": zone, "event": "entered", "timestamp": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(), "dwell_seconds": None})
        elif prev_zone != zone:
            dwell = round(now - self._zone_entry_ts.get(track_id, now), 1)
            self._events.append({"track_id": track_id, "zone": prev_zone, "event": "exited", "timestamp": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(), "dwell_seconds": dwell})
            self._events.append({"track_id": track_id, "zone": zone, "event": "entered", "timestamp": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(), "dwell_seconds": None})
            self._current_zone[track_id]  = zone
            self._zone_entry_ts[track_id] = now

    def flush_lost_tracks(self, active_ids, now):
        lost = set(self._current_zone.keys()) - active_ids
        for tid in lost:
            zone  = self._current_zone.pop(tid)
            dwell = round(now - self._zone_entry_ts.pop(tid, now), 1)
            self._events.append({"track_id": tid, "zone": zone, "event": "exited", "timestamp": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(), "dwell_seconds": dwell})

    def pop_events(self):
        events, self._events = self._events, []
        return events

class CameraWorker(threading.Thread):
    def __init__(self, name, url, target_fps, logger, blob_emitter, zones=None, counting_line_y=0.5, use_small_model=False, store_open_time=None):
        super().__init__(daemon=True)
        self.name            = name
        self.url             = url
        self.target_fps      = target_fps
        self.frame_interval  = 1 / target_fps
        self.running         = True
        self.cap             = None
        self.logger          = logger
        self.blob_emitter    = blob_emitter
        self.zones           = zones or {"full_frame": (0, 0, 1, 1)}
        self.counting_line_y = counting_line_y
        self.store_open_time = store_open_time
        from ultralytics import YOLO
        model_name = "yolov8s.pt" if use_small_model else "yolov8n.pt"
        self.model = YOLO(model_name)
        self.logger.info(f"{self.name} loaded {model_name}")
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self.tracker          = DeepSort(max_age=DEEPSORT_MAX_AGE)
        self.zone_tracker     = ZoneTracker()
        self.track_positions  = {}
        self.zone_buffer      = collections.defaultdict(int)
        self.count_in         = 0
        self.count_out        = 0
        self.detection_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)

    def connect(self):
        while True:
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if self.cap.isOpened():
                self.logger.info(f"{self.name} connected")
                return
            self.logger.warning(f"{self.name} connection failed, retrying...")
            time.sleep(2)

    def _is_staff_period(self):
        if self.store_open_time is None:
            return False
        return (datetime.now() - self.store_open_time).total_seconds() < STAFF_IGNORE_MINUTES * 60

    def _get_zone(self, nx, ny):
        for zone_name, (x1, y1, x2, y2) in self.zones.items():
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                return zone_name
        return "unknown"

    def _update_counts(self, track_id, ny):
        prev_y = self.track_positions.get(track_id)
        self.track_positions[track_id] = ny
        near_line = abs(ny - self.counting_line_y) < 0.08
        if near_line:
            self.zone_buffer[track_id] += 1
        else:
            if self.zone_buffer[track_id] >= ENTRY_ZONE_BUFFER and prev_y is not None:
                if prev_y < self.counting_line_y and ny > self.counting_line_y:
                    self.count_in += 1
                    self.zone_buffer[track_id] = 0
                    return "entry"
                elif prev_y > self.counting_line_y and ny < self.counting_line_y:
                    self.count_out += 1
                    self.zone_buffer[track_id] = 0
                    return "exit"
            self.zone_buffer[track_id] = 0
        return None

    def _detect(self, frame):
        h, w = frame.shape[:2]
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, classes=[0], verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"bbox": [x1/w, y1/h, x2/w, y2/h], "conf": round(float(box.conf[0]), 3)})
        return detections

    def _smooth(self, current):
        self.detection_buffer.append(current)
        if len(self.detection_buffer) < SMOOTHING_WINDOW:
            return current
        return current if sum(len(d) for d in self.detection_buffer) / SMOOTHING_WINDOW >= 0.5 else []

    def run(self):
        self.connect()
        last_frame_time = 0
        fail_count = 0
        frame_id   = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                fail_count += 1
                if fail_count >= 5:
                    self.logger.warning(f"{self.name} stream lost, reconnecting...")
                    self.cap.release()
                    self.connect()
                    fail_count = 0
                time.sleep(0.5)
                continue
            fail_count = 0
            now = time.time()
            if now - last_frame_time < self.frame_interval:
                continue
            last_frame_time = now
            frame_id += 1
            if self._is_staff_period():
                continue
            dets = self._smooth(self._detect(frame))
            h, w = frame.shape[:2]
            ds_input = []
            for d in dets:
                x1n, y1n, x2n, y2n = d["bbox"]
                ds_input.append(([x1n*w, y1n*h, (x2n-x1n)*w, (y2n-y1n)*h], d["conf"], "person"))
            tracks     = self.tracker.update_tracks(ds_input, frame=frame)
            active_ids = set()
            crossings  = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                l, t, r, b = track.to_ltrb()
                nx = ((l+r)/2) / w
                ny = ((t+b)/2) / h
                active_ids.add(tid)
                self.zone_tracker.update(tid, self._get_zone(nx, ny), now)
                crossing = self._update_counts(tid, ny)
                if crossing:
                    crossings.append(crossing)
            self.zone_tracker.flush_lost_tracks(active_ids, now)
            zone_events = self.zone_tracker.pop_events()
            if zone_events or active_ids or crossings:
                self.blob_emitter.enqueue({
                    "camera":      self.name,
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                    "frame_id":    frame_id,
                    "people_now":  len(active_ids),
                    "counts":      {"in": self.count_in, "out": self.count_out, "current": max(0, self.count_in - self.count_out)},
                    "crossings":   crossings,
                    "zone_events": zone_events,
                })
            if frame_id % 50 == 0:
                self.logger.info(f"{self.name} | frame={frame_id} | people={len(active_ids)} | in={self.count_in} out={self.count_out}")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
