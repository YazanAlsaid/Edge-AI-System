import cv2
import tempfile

from inference.inference_pipeline import inferencePipeline


def run_inference(
    input_video_path: str,
    hef_path: str,
    labels_path: str,
    config_path: str,
    enable_tracking: bool = True,
    enable_depth: bool = False,
):
    """
    Run inference on the entire video.

    Returns:
        dict {
            "yolo_video": str,
            "depth_video": str | None,
            "frames": int
        }
    """

    # ---------------- Initialize stream ----------------
    stream = inferencePipeline(
        hef_path=hef_path,
        labels_path=labels_path,
        config_path=config_path,
        enable_tracking=enable_tracking,
    )
    stream.reset()

    if enable_depth:
        stream.enable_depth_estimation(True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 10
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # ---------------- Output videos ----------------
    tmp_yolo = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    yolo_video_path = tmp_yolo.name
    tmp_yolo.close()

    yolo_writer = cv2.VideoWriter(
        yolo_video_path, fourcc, fps, (width, height)
    )

    depth_video_path = None
    depth_writer = None

    if enable_depth:
        tmp_depth = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        depth_video_path = tmp_depth.name
        tmp_depth.close()

        depth_writer = cv2.VideoWriter(
            depth_video_path, fourcc, fps, (width, height)
        )

    frame_count = 0

    # ---------------- Main loop ----------------
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = stream.process_frame(frame)

            # YOLO video
            yolo_bgr = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            yolo_writer.write(yolo_bgr)

            # Depth video
            if enable_depth and result["depth"] is not None:
                depth_norm = cv2.normalize(
                    result["depth"], None, 0, 255, cv2.NORM_MINMAX
                ).astype("uint8")

                depth_colored = cv2.applyColorMap(
                    depth_norm, cv2.COLORMAP_INFERNO
                )

                depth_writer.write(depth_colored)

            frame_count += 1

    finally:
        cap.release()
        yolo_writer.release()
        if depth_writer:
            depth_writer.release()
        stream.stop()

    return {
        "yolo_video": yolo_video_path,
        "depth_video": depth_video_path,
        "frames": frame_count,
    }
