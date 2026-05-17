# Auris Edge Configuration
# Supports 6-7 RTSP streams perfectly on an Intel N100 (Alder Lake-N)
# because motion detection uses <5% CPU per stream at 2 FPS.

CAMERAS = {
    # Replace with your actual NVR / IP Camera RTSP URLs
    "cam1": {"url": "rtsp://admin:password@192.168.1.101:554/stream1", "fps": 2},
    "cam2": {"url": "rtsp://admin:password@192.168.1.102:554/stream1", "fps": 2},
    "cam3": {"url": "rtsp://admin:password@192.168.1.103:554/stream1", "fps": 2},
    "cam4": {"url": "rtsp://admin:password@192.168.1.104:554/stream1", "fps": 2},
    "cam5": {"url": "rtsp://admin:password@192.168.1.105:554/stream1", "fps": 2},
    "cam6": {"url": "rtsp://admin:password@192.168.1.106:554/stream1", "fps": 2},
    
    # If a camera goes offline, the edge_worker will gracefully retry 
    # connection every 5 seconds without crashing the other 5 threads.
}
