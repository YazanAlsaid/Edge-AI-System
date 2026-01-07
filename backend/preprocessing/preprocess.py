import cv2
import numpy as np
from typing import Optional, Callable, List
import queue


####################################################################
# Preprocessing of Network Input
####################################################################

def default_preprocess(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    """
    Resize image with unchanged aspect ratio using padding.

    Args:
        image (np.ndarray): Input image.
        model_w (int): Model input width.
        model_h (int): Model input height.

    Returns:
        np.ndarray: Preprocessed and padded image.
    """
    img_h, img_w, _ = image.shape[:3]

    scale = min(model_w / img_w, model_h / img_h)
    new_img_w = int(img_w * scale)
    new_img_h = int(img_h * scale)

    image = cv2.resize(
        image,
        (new_img_w, new_img_h),
        interpolation=cv2.INTER_CUBIC,
    )

    padded_image = np.full(
        (model_h, model_w, 3),
        (114, 114, 114),
        dtype=np.uint8,
    )

    x_offset = (model_w - new_img_w) // 2
    y_offset = (model_h - new_img_h) // 2

    padded_image[
        y_offset:y_offset + new_img_h,
        x_offset:x_offset + new_img_w,
    ] = image

    return padded_image


def preprocess_for_streamlit(
    frame: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Optimized preprocessing for Streamlit using a single frame.

    Args:
        frame (np.ndarray): BGR frame from OpenCV.
        width (int): Model input width.
        height (int): Model input height.

    Returns:
        np.ndarray: Preprocessed frame ready for inference.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed = default_preprocess(rgb_frame, width, height)
    return processed


def batch_preprocess(
    cap: cv2.VideoCapture,
    batch_size: int,
    input_queue: queue.Queue,
    width: int,
    height: int,
):
    """
    Batch preprocessing for offline video inference.
    """
    frames = []
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = default_preprocess(rgb, width, height)
        processed_frames.append(processed)

        if len(frames) == batch_size:
            input_queue.put((frames, processed_frames))
            frames = []
            processed_frames = []

    input_queue.put(None)
