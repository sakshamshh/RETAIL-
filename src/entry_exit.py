from datetime import datetime

class EntryExitCounter:
    def __init__(self, line_y_ratio=0.5):
        self.line_y_ratio = line_y_ratio
        self.track_positions = {}
        self.entries = 0
        self.exits = 0
        self.counted_ids = set()

    def update(self, tracks, frame_height):
        line_y = int(frame_height * self.line_y_ratio)

        for (tid, x1, y1, x2, y2) in tracks:
            cy = (y1 + y2) // 2

            if tid in self.track_positions:
                prev_cy = self.track_positions[tid]

                if tid not in self.counted_ids:
                    if prev_cy < line_y and cy >= line_y:
                        self.entries += 1
                        self.counted_ids.add(tid)
                    elif prev_cy > line_y and cy <= line_y:
                        self.exits += 1
                        self.counted_ids.add(tid)

            self.track_positions[tid] = cy

        return {
            "entries": self.entries,
            "exits": self.exits,
            "net": self.entries - self.exits
        }

counters = {}

def get_counter(cam_name):
    if cam_name not in counters:
        counters[cam_name] = EntryExitCounter()
    return counters[cam_name]
