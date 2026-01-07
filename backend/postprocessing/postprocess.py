import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from common.toolbox import (
    id_to_color,
    normalize_depths,
    depth_median_lower_half,
)

from .detection_utils import (
    extract_detections,
    draw_detections,
    compute_roi,
    draw_roi,
)


def inference_result_handler(
    original_frame: np.ndarray,
    infer_results: list,
    labels: List[str],
    config_data: Dict,
    tracker: Optional = None,
    draw_trail: bool = False,
    depth_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Main postprocessing function.
    """

    # Compute region of interest once
    roi_cfg = config_data.get("roi", {})
    roi = compute_roi(
        original_frame.shape,
        roi_cfg.get("x_min", 0.20),
        roi_cfg.get("x_max", 0.80),
        roi_cfg.get("y_min", 0.45),
        roi_cfg.get("y_max", 1.00),
       )

    draw_roi(original_frame, roi)

    # Extract detections from model output
    detections = extract_detections(original_frame, infer_results, config_data,roi)

    depths = []

    if depth_map is not None:
        for box in detections["detection_boxes"]:
            depth_val = depth_median_lower_half(depth_map, box)
            depths.append(depth_val)
    else:
        depths = [128.0] * detections["num_detections"]

    detections["detection_depths"] = depths
    detections["detection_depths_norm"] = normalize_depths(depths)

    # Draw detections and tracking results
    annotated_frame = draw_detections(
        detections,
        original_frame,
        labels,
        tracker,
        draw_trail,
    )

    return annotated_frame
