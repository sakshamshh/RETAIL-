import os, json, requests
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SERVER = os.getenv("CLOUD_ENDPOINT", "https://retailiq.centralindia.cloudapp.azure.com").replace("/api/blobs", "")
GROQ_KEY = os.getenv("GROQ_API_KEY")
STORE = os.getenv("STORE_NAME", "Your Store")
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def fetch(path):
    try:
        r = requests.get(SERVER + path, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Failed to fetch {path}: {e}")
        return None

def build_summary(today, hourly, zones, live):
    lines = []
    date = today.get("date", datetime.now().strftime("%Y-%m-%d")) if today else "today"
    lines.append(f"Date: {date}")
    lines.append(f"Store: {STORE}")
    if today and today.get("cameras"):
        total_in = sum(c.get("total_in", 0) for c in today["cameras"])
        total_out = sum(c.get("total_out", 0) for c in today["cameras"])
        current = sum(c.get("current", 0) for c in today["cameras"])
        lines.append(f"\nFootfall Summary:")
        lines.append(f"- Total visitors entered: {total_in}")
        lines.append(f"- Total visitors exited: {total_out}")
        lines.append(f"- Currently in store: {current}")
    if hourly and hourly.get("hourly"):
        by_hour = {}
        for h in hourly["hourly"]:
            hour = h["_id"]["hour"]
            entries = h.get("entries", 0)
            by_hour[hour] = by_hour.get(hour, 0) + entries
        if by_hour:
            peak_hour = max(by_hour, key=by_hour.get)
            slow_hour = min(by_hour, key=by_hour.get)
            lines.append(f"\nHourly Traffic:")
            lines.append(f"- Peak hour: {peak_hour}:00 with {by_hour[peak_hour]} entries")
            lines.append(f"- Slowest hour: {slow_hour}:00 with {by_hour[slow_hour]} entries")
    if zones and zones.get("zones"):
        lines.append(f"\nZone Activity:")
        for z in zones["zones"][:8]:
            zone_name = z["_id"].get("zone", "unknown")
            cam = z["_id"].get("camera", "")
            lines.append(f"- {cam}/{zone_name}: {z['count']} detections")
    return "\n".join(lines)

def generate_report(summary):
    client = Groq(api_key=GROQ_KEY)
    prompt = store_context = f"""
Store Profile:
- Name: {os.getenv('STORE_NAME', 'Unknown')}
- Type: {os.getenv('STORE_TYPE', 'retail')}
- Location: {os.getenv('STORE_AREA', 'India')}
- Staff count: {os.getenv('STORE_STAFF', '2-3')}
- Opening hours: {os.getenv('STORE_OPEN', '10:00')} to {os.getenv('STORE_CLOSE', '21:00')}
- Average basket size: ₹{os.getenv('AVG_BASKET', 'unknown')}
- Weekly revenue target: ₹{os.getenv('WEEKLY_TARGET', 'unknown')}
- Busy days: {os.getenv('BUSY_DAYS', 'weekends')}
- Slow days: {os.getenv('SLOW_DAYS', 'tuesday')}
- Competition nearby: {os.getenv('NEARBY_COMPETITION', 'unknown')}
- Owner notes: {os.getenv('STORE_NOTES', 'none')}
"""

prompt = f"""You are RetailIQ, an AI business advisor for Indian retail stores.

{store_context}

Today's analytics data:
{summary}

Write a daily business report for the store owner. Be specific to THIS store — 
mention the store type, location context, staff count, and revenue targets in your advice.
Write in simple English mixed with natural Hindi words.
Be direct like a trusted advisor, not a corporate report.

Structure exactly like this:

TODAY'S SUMMARY
[2-3 sentences specific to this store's day]

PEAK HOURS ANALYSIS
[Specific advice for THIS store type during peak hours]

ZONE INSIGHTS
[Product/display recommendations specific to THIS store type]

WHAT TO WATCH
[1-2 concerns based on today's data AND store context]

ACTION ITEMS FOR TOMORROW
[3 specific numbered actions — reference staff count, store type, targets]

THIS WEEK'S FOCUS
[One strategic suggestion tied to weekly target of ₹{os.getenv('WEEKLY_TARGET', 'your target')}]

Keep under 400 words. Be direct. No fluff."""
