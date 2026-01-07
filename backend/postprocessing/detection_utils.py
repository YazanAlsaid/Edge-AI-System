import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Set
from collections import deque

from common.toolbox import id_to_color, depth_to_zone_and_color


def denormalize_and_rm_pad(
    box: list,
    size: int,
    padding_length: int,
    input_height: int,
    input_width: int,
):
    """
    Denormalize bounding box coordinates and remove padding.
    """
    for i, x in enumerate(box):
        box[i] = int(x * size)

        if (input_width != size) and (i % 2 != 0):
            box[i] -= padding_length

        if (input_height != size) and (i % 2 == 0):
            box[i] -= padding_length

    return box


def extract_detections(image: np.ndarray, detections: list, config_data,roi: tuple) -> dict:
    """
    Extract detections from model output.
    """

    visualization_params = config_data["visualization_params"]
    score_threshold = visualization_params.get("score_thres", 0.5)
    max_boxes = visualization_params.get("max_boxes_to_draw", 50)

    img_height, img_width = image.shape[:2]
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)
    """
    roi_cfg = config_data.get("roi", {})
    roi = compute_roi(
        image.shape,
        roi_cfg.get("x_min", 0.20),
        roi_cfg.get("x_max", 0.80),
        roi_cfg.get("y_min", 0.45),
        roi_cfg.get("y_max", 1.00),
    )
    """
    all_detections = []

    for class_id, detection in enumerate(detections):
        for det in detection:
            bbox, score = det[:4], det[4]

            if score >= score_threshold:
                denorm_bbox = denormalize_and_rm_pad(
                    bbox,
                    size,
                    padding_length,
                    img_height,
                    img_width,
                )

                if box_center_in_roi(denorm_bbox, roi):
                   all_detections.append((score, class_id, denorm_bbox))

    all_detections.sort(reverse=True, key=lambda x: x[0])
    top_detections = all_detections[:max_boxes]

    scores, class_ids, boxes = (
        zip(*top_detections) if top_detections else ([], [], [])
    )

    return {
        "detection_boxes": list(boxes),
        "detection_classes": list(class_ids),
        "detection_scores": list(scores),
        "num_detections": len(top_detections),
    }


def draw_detection(
    image: np.ndarray,
    box: list,
    labels: list,
    score: float,
    color: tuple,
    track: bool = False,
) -> None:
    """
    Draw a single detection on the image.
    """

    ymin, xmin, ymax, xmax = map(int, box)

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    top_text = (
        f"{labels[0]}: {score:.1f}%"
        if not track or len(labels) == 2
        else f"{score:.1f}%"
    )

    bottom_text = None
    if track:
        bottom_text = labels[1] if len(labels) == 2 else labels[0]

    cv2.putText(
        image,
        top_text,
        (xmin + 4, ymin + 20),
        font,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        top_text,
        (xmin + 4, ymin + 20),
        font,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if bottom_text:
        pos = (xmax - 50, ymax - 6)

        cv2.putText(
            image,
            bottom_text,
            pos,
            font,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            bottom_text,
            pos,
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_roi(image: np.ndarray, roi: dict) -> None:
    """
    Draw ROI rectangle on the image.
    """
    cv2.rectangle(
        image,
        (roi["x1"], roi["y1"]),
        (roi["x2"], roi["y2"]),
        (0, 255, 0),
        2,
    )

def draw_detections(
    detections: dict,
    img_out: np.ndarray,
    labels,
    tracker=None,
    draw_trail=False,
):

    #Draw detections or tracking results on the image.


    boxes = detections["detection_boxes"]
    scores = detections["detection_scores"]
    num_detections = detections["num_detections"]
    classes = detections["detection_classes"]

    if tracker:
        dets_for_tracker = []

        for idx in range(num_detections):
            box = boxes[idx]
            score = scores[idx]
            dets_for_tracker.append([*box, score])

        if not dets_for_tracker:
            return img_out

        online_targets = tracker.update(np.array(dets_for_tracker))

        for track in online_targets:
            track_id = track.track_id
            x1, y1, x2, y2 = track.tlbr
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])

            best_idx = find_best_matching_detection_index(
                track.tlbr,
                boxes,
            )

            depths = detections.get("detection_depths", [])

            if best_idx is not None and best_idx < len(depths):
               depth_val = depths[best_idx]
               zone, color = depth_to_zone_and_color(depth_val)
               label = f"{labels[classes[best_idx]]} | {zone}"
            else:
              color = (128, 128, 128)
              zone = "UNKNOWN"

            draw_detection(
                img_out,
                [xmin, ymin, xmax, ymax],
                [label, f"ID {track_id}"],
                track.score * 100.0,
                color,
                track=True,
            )

    else:
        for idx in range(num_detections):
            color = tuple(id_to_color(classes[idx]).tolist())
            draw_detection(
                img_out,
                boxes[idx],
                [labels[classes[idx]]],
                scores[idx] * 100.0,
                color,
            )

    return img_out

def find_best_matching_detection_index(track_box, detection_boxes):
    """
    Find the detection index with the highest IoU to the tracking box.
    """
    best_iou = 0
    best_idx = -1

    for i, det_box in enumerate(detection_boxes):
        iou = compute_iou(track_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    return best_idx if best_idx != -1 else None


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union between two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return inter / (areaA + areaB - inter + 1e-5)


def compute_roi(
    image_shape: tuple,
    x_min_ratio: float = 0.20,
    x_max_ratio: float = 0.80,
    y_min_ratio: float = 0.45,
    y_max_ratio: float = 1.00,
) -> dict:
    """
    Compute pixel-based ROI from image shape and relative ratios.
    """
    h, w = image_shape[:2]

    return {
        "x1": int(x_min_ratio * w),
        "x2": int(x_max_ratio * w),
        "y1": int(y_min_ratio * h),
        "y2": int(y_max_ratio * h),
    }


def box_center_in_roi(box: list, roi: dict) -> bool:
    """
    Check whether the center of a bounding box lies inside the ROI.
    """
    ymin, xmin, ymax, xmax = box
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5

    return (
        roi["x1"] <= cx <= roi["x2"]
        and roi["y1"] <= cy <= roi["y2"]
    )


