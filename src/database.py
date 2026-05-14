import sqlite3
import os
from datetime import datetime

DB_PATH = "data/retailiq.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS traffic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera TEXT,
            timestamp TEXT,
            hour INTEGER,
            day TEXT,
            is_weekend INTEGER,
            people_count INTEGER,
            entries INTEGER,
            exits INTEGER,
            net INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            camera TEXT,
            message TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_traffic(camera, people_count, entries, exits, net):
    now = datetime.now()
    is_weekend = 1 if now.strftime("%A") in ["Saturday", "Sunday"] else 0
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO traffic (camera, timestamp, hour, day, is_weekend, people_count, entries, exits, net)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (camera, now.isoformat(), now.hour, now.strftime("%A"), is_weekend, people_count, entries, exits, net))
    conn.commit()
    conn.close()

def save_alert(camera, message):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO alerts (timestamp, camera, message) VALUES (?, ?, ?)",
              (datetime.now().isoformat(), camera, message))
    conn.commit()
    conn.close()

def get_hourly_traffic(camera=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if camera:
        c.execute("SELECT hour, AVG(people_count) FROM traffic WHERE camera=? GROUP BY hour ORDER BY hour", (camera,))
    else:
        c.execute("SELECT hour, AVG(people_count) FROM traffic GROUP BY hour ORDER BY hour")
    rows = c.fetchall()
    conn.close()
    return rows

def get_daily_summary():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT day, AVG(people_count), SUM(entries), SUM(exits) FROM traffic GROUP BY day")
    rows = c.fetchall()
    conn.close()
    return rows

def get_today_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("""
        SELECT SUM(entries), SUM(exits), MAX(people_count), AVG(people_count)
        FROM traffic WHERE timestamp LIKE ?
    """, (f"{today}%",))
    row = c.fetchone()
    conn.close()
    return {
        "total_entries": row[0] or 0,
        "total_exits": row[1] or 0,
        "peak_count": row[2] or 0,
        "avg_count": round(row[3] or 0, 1)
    }
