import numpy as np
import logging

class AutoCalibrator:
    """
    The Azure Mathematical Brain for Zero-Click Auto-Calibration.
    Analyzes blob tracking data over 48 hours to automatically deduce Entrance Lines and Hotzones.
    """
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def find_entrance_line(self, blob_paths):
        """
        Calculates the Y-coordinate of the main entrance by analyzing where blob tracks begin/end.
        blob_paths: list of lists, where each inner list is a track of (x, y) coordinates for a single blob.
        """
        if not blob_paths:
            return None

        # Extract all starting and ending Y coordinates of every track
        terminal_y_points = []
        for path in blob_paths:
            if len(path) > 1:
                terminal_y_points.append(path[0][1])  # Start Y
                terminal_y_points.append(path[-1][1]) # End Y

        if not terminal_y_points:
            return None

        # Mathematically, the entrance is where the highest density of paths start/end.
        # We use a simple 1D histogram clustering to find the dominant Y-band.
        hist, bin_edges = np.histogram(terminal_y_points, bins=20)
        peak_bin_idx = np.argmax(hist)
        
        # The calculated entrance line is the center of the densest bin
        entrance_y = (bin_edges[peak_bin_idx] + bin_edges[peak_bin_idx + 1]) / 2.0
        
        self.logger.info(f"Auto-Calibration complete. Entrance line detected at Y={entrance_y:.2f}")
        return entrance_y

    def find_hotzones(self, blob_paths, frame_width, frame_height, grid_size=50):
        """
        Calculates Hotzones (areas of high dwell time) by creating a spatial heatmap.
        Returns a list of bounding boxes [x1, y1, x2, y2] representing mathematical zones.
        """
        if not blob_paths:
            return []

        # Create a 2D spatial grid (heatmap)
        grid_w = frame_width // grid_size + 1
        grid_h = frame_height // grid_size + 1
        heatmap = np.zeros((grid_h, grid_w))

        # Accumulate dwell time (each point in a path represents time spent)
        for path in blob_paths:
            for x, y in path:
                gx, gy = int(x // grid_size), int(y // grid_size)
                if 0 <= gx < grid_w and 0 <= gy < grid_h:
                    heatmap[gy, gx] += 1

        # Identify hot cells (cells with activity > 90th percentile)
        threshold = np.percentile(heatmap[heatmap > 0], 90) if np.any(heatmap > 0) else 0
        hotzones = []
        
        if threshold > 0:
            for gy in range(grid_h):
                for gx in range(grid_w):
                    if heatmap[gy, gx] >= threshold:
                        # Convert grid cell back to pixel bounding box
                        x1 = gx * grid_size
                        y1 = gy * grid_size
                        hotzones.append([x1, y1, x1 + grid_size, y1 + grid_size])

        self.logger.info(f"Auto-Calibration found {len(hotzones)} high-density Hotzones.")
        return hotzones
