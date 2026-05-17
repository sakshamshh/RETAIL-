import json, time, queue, threading, sqlite3, logging, os, requests
from datetime import datetime, timezone

CLOUD_ENDPOINT = os.getenv("CLOUD_ENDPOINT", "http://98.70.41.191:8000/api/blobs")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "")
BUFFER_DB_PATH = os.getenv("BUFFER_DB_PATH", "data/blob_buffer.db")
UPLOAD_TIMEOUT = 5
RETRY_INTERVAL = 15
BATCH_SIZE = 50

class BlobEmitter:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self._queue = queue.Queue(maxsize=1000)
        os.makedirs(os.path.dirname(BUFFER_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(BUFFER_DB_PATH)
        conn.execute("CREATE TABLE IF NOT EXISTS blob_buffer (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL, created TEXT NOT NULL)")
        conn.commit()
        conn.close()
        threading.Thread(target=self._upload_loop, daemon=True).start()
        threading.Thread(target=self._retry_loop, daemon=True).start()
        self.logger.info("BlobEmitter started")

    def enqueue(self, blob):
        try:
            self._queue.put_nowait(blob)
        except queue.Full:
            self.logger.warning("Queue full")

    def _post(self, blobs):
        try:
            headers = {"Content-Type": "application/json"}
            if CLOUD_API_KEY:
                headers["X-API-Key"] = CLOUD_API_KEY
            resp = requests.post(CLOUD_ENDPOINT, json={"blobs": blobs}, headers=headers, timeout=UPLOAD_TIMEOUT)
            return resp.status_code in (200, 201, 202)
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Upload failed: {e}")
            return False

    def _buffer_blob(self, blob):
        conn = sqlite3.connect(BUFFER_DB_PATH)
        conn.execute("INSERT INTO blob_buffer (payload, created) VALUES (?, ?)", (json.dumps(blob), datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()

    def _upload_loop(self):
        while True:
            blob = self._queue.get()
            if not self._post([blob]):
                self._buffer_blob(blob)

    def _retry_loop(self):
        while True:
            time.sleep(RETRY_INTERVAL)
            conn = sqlite3.connect(BUFFER_DB_PATH)
            rows = conn.execute("SELECT id, payload FROM blob_buffer ORDER BY id ASC LIMIT ?", (BATCH_SIZE,)).fetchall()
            conn.close()
            if not rows:
                continue
            ids = [r[0] for r in rows]
            blobs = [json.loads(r[1]) for r in rows]
            if self._post(blobs):
                conn = sqlite3.connect(BUFFER_DB_PATH)
                conn.execute(f"DELETE FROM blob_buffer WHERE id IN ({','.join('?'*len(ids))})", ids)
                conn.commit()
                conn.close()
                self.logger.info(f"Flushed {len(ids)} blobs")
