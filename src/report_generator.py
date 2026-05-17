import os, json, requests
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SERVER = os.getenv("CLOUD_ENDPOINT", "https://Auris.centralindia.cloudapp.azure.com").replace("/api/blobs", "")
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
    store_context = f"""
Store Profile:
- Name: {os.getenv('STORE_NAME', 'Unknown')}
- Type: {os.getenv('STORE_TYPE', 'retail')}
- Location: {os.getenv('STORE_AREA', 'India')}
- Staff count: {os.getenv('STORE_STAFF', '2-3')}
- Opening hours: {os.getenv('STORE_OPEN', '10:00')} to {os.getenv('STORE_CLOSE', '21:00')}
- Average basket size: {os.getenv('AVG_BASKET', 'unknown')}
- Weekly revenue target: {os.getenv('WEEKLY_TARGET', 'unknown')}
- Busy days: {os.getenv('BUSY_DAYS', 'weekends')}
- Slow days: {os.getenv('SLOW_DAYS', 'tuesday')}
- Competition nearby: {os.getenv('NEARBY_COMPETITION', 'unknown')}
- Owner notes: {os.getenv('STORE_NOTES', 'none')}
"""

    prompt = f"""You are Auris, an AI business advisor for Indian retail stores.

{store_context}

Today's analytics data:
{summary}

Analyze the data and provide actionable business advice specific to this store.
Write in simple English mixed with natural Hindi words. Be direct like a trusted advisor.

CRITICAL: You MUST output ONLY valid JSON. Do not include markdown code blocks or any other text.
The JSON must have this exact structure:
{{
  "sections": [
    {{ "title": "TODAY'S SUMMARY", "body": "2-3 sentences summarizing the day." }},
    {{ "title": "PEAK HOURS ANALYSIS", "body": "Specific advice during peak hours." }},
    {{ "title": "ZONE INSIGHTS", "body": "Product/display recommendations based on zone data." }},
    {{ "title": "WHAT TO WATCH", "body": "1-2 concerns based on today's data." }},
    {{ "title": "ACTION ITEMS FOR TOMORROW", "body": "1. First action
2. Second action
3. Third action" }}
  ]
}}
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"Report generation error: {e}")
        return {"raw": "Failed to generate report structure. Please try again."}
