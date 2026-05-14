import sys
sys.path.insert(0, '.')
from src.database import init_db
import sqlite3
from datetime import datetime

init_db()
conn = sqlite3.connect('data/retailiq.db')
c = conn.cursor()

data = [
    (9, 3, 8), (10, 5, 12), (11, 8, 15), (12, 12, 20),
    (13, 10, 18), (14, 9, 16), (15, 11, 19), (16, 13, 22),
    (17, 15, 25), (18, 11, 18), (19, 7, 12), (20, 4, 8)
]

for h, avg, ent in data:
    ts = datetime.now().strftime(f'%Y-%m-%dT{h:02d}:00:00')
    c.execute(
        'INSERT INTO traffic (camera,timestamp,hour,day,is_weekend,people_count,entries,exits,net) VALUES (?,?,?,?,?,?,?,?,?)',
        ('cam1', ts, h, 'Monday', 0, avg, ent, ent-2, 2)
    )

conn.commit()
conn.close()
print('Seed data inserted!')
