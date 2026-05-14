from datetime import datetime, time
import collections
import os
from dotenv import load_dotenv

load_dotenv()

MAX_CAPACITY = int(os.getenv("MAX_CAPACITY", 50))
_open = os.getenv("STORE_OPEN", "09:00").split(":")
_close = os.getenv("STORE_CLOSE", "21:00").split(":")
STORE_OPEN = time(int(_open[0]), int(_open[1]))
STORE_CLOSE = time(int(_close[0]), int(_close[1]))

class RetailAnalytics:
    def __init__(self):
        self.current_occupancy = 0
        self.hourly_counts = collections.defaultdict(int)
        self.last_alerts = {}

    def is_store_open(self):
        now = datetime.now().time()
        return STORE_OPEN <= now <= STORE_CLOSE

    def should_alert(self, key, cooldown=60):
        now = datetime.now().timestamp()
        if key not in self.last_alerts or now - self.last_alerts[key] > cooldown:
            self.last_alerts[key] = now
            return True
        return False

    def update(self, cam_name, people_count):
        now = datetime.now()
        hour = now.hour
        day = now.strftime("%A")
        is_weekend = day in ["Saturday", "Sunday"]

        self.hourly_counts[hour] += people_count
        self.current_occupancy = people_count

        alerts = []

        if people_count > MAX_CAPACITY:
            if self.should_alert("overcapacity"):
                alerts.append(f"OVERCAPACITY: {people_count} people in store!")

        if people_count > MAX_CAPACITY * 0.8:
            if self.should_alert("crowding"):
                alerts.append(f"CROWDING ALERT: {people_count} people detected")

        peak_hour = max(self.hourly_counts, key=self.hourly_counts.get)
        if hour == peak_hour:
            if self.should_alert("peak_hour", cooldown=300):
                alerts.append(f"PEAK HOUR: {hour}:00 is busiest so far")

        if not self.is_store_open() and people_count > 0:
            if self.should_alert("after_hours"):
                alerts.append(f"AFTER HOURS: {people_count} people detected!")

        period = "WEEKEND" if is_weekend else "WEEKDAY"

        stats = {
            "type": "analytics",
            "camera": cam_name,
            "people_count": people_count,
            "occupancy": self.current_occupancy,
            "peak_hour": f"{peak_hour}:00",
            "period": period,
            "store_open": self.is_store_open(),
            "alerts": alerts,
            "hour": hour
        }

        return stats

analytics = RetailAnalytics()
