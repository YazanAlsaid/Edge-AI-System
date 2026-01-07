from typing import List, Generator,Dict, Any
import json
import os
import sys
import numpy as np
import cv2
import time


# ==================================================
# JSON / Config Utilities
# ==================================================

def load_json_file(path: str) -> Dict[str, Any]:
    """
    Loads and parses a JSON file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON format in file '{path}': {e.msg}",
                e.doc,
                e.pos
            )
    return data


def get_labels(labels_path: str) -> list:
    """
    Load labels from a file.
    """
    with open(labels_path, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()
    return class_names


# ==================================================
# Input / Video Utilities
# ==================================================

def init_input_source(input_path: str):
    if not any(input_path.lower().endswith(ext) for ext in ('.mp4', '.avi', '.mov', '.mkv')):
        raise ValueError("Only video files are supported (.mp4, .avi, .mov, .mkv)")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    return cap


# ==================================================
# Color Utilities
# ==================================================

def generate_color(class_id: int) -> tuple:
    """
    Generate a unique color for a given class ID.
    """
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())


def id_to_color(idx):
    np.random.seed(idx)
    return np.random.randint(0, 255, size=3, dtype=np.uint8)

def depth_to_zone_and_color(depth_val: float):
    """
    Map relative MiDaS depth value to proximity zone and BGR color.

    depth_val: value in range [0, 255]
    Lower value means closer.
    """
    if depth_val < 85:
        return "NEAR", (0, 0, 255)
    elif depth_val < 160:
        return "MID", (0, 255, 255)
    else:
        return "FAR", (0, 255, 0)
# ==================================================
# Depth / Utilities
# ==================================================
def depth_median_lower_half(depth_map: np.ndarray, box: list) -> float:
    """
    Compute a robust relative depth value for an object using:
    - lower half of the bounding box
    - median aggregation

    Returns a value in [0, 255] (uint8 MiDaS depth space).
    Lower value = closer.
    """
    x1, y1, x2, y2 = map(int, box)

    h, w = depth_map.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return 128.0

    y_mid = y1 + (y2 - y1) // 2
    depth_crop = depth_map[y_mid:y2, x1:x2]

    if depth_crop.size == 0:
        return 128.0

    return float(np.median(depth_crop))

def normalize_depths(depths: list) -> list:
    """
    Normalize depth values per frame to range [0, 1].
    0 = nearest, 1 = farthest
    """
    if not depths:
        return []

    d_min = min(depths)
    d_max = max(depths)

    if d_max > d_min:
        return [(d - d_min) / (d_max - d_min) for d in depths]
    else:
        # all depths equal -> neutral value
        return [0.5] * len(depths)


# ==================================================
# Frame Rate Tracker
# ==================================================

class FrameRateTracker:
    def __init__(self):
        self._count = 0
        self._start_time = None

    def start(self) -> None:
        self._start_time = time.time()

    def increment(self, n: int = 1) -> None:
        self._count += n

    @property
    def count(self) -> int:
        return self._count

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def fps(self) -> float:
        elapsed = self.elapsed
        return self._count / elapsed if elapsed > 0 else 0.0

    def frame_rate_summary(self) -> str:
        return f"Processed {self.count} frames at {self.fps:.2f} FPS"
