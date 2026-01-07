import cv2
import os
import time
import queue
import numpy as np
from typing import Optional, Callable, Any


def visualize_offline(output_queue: queue.Queue, cap: cv2.VideoCapture, save_stream_output: bool, output_dir: str,
              callback: Callable, fps_tracker: Optional["FrameRateTracker"] = None, 
              side_by_side: bool = False, disable_display: bool = False) -> None:

    image_id = 0
    out = None

    if cap is not None:
        if not disable_display:
            cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if save_stream_output:
            base_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            base_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = base_width * 2 if side_by_side else base_width
            frame_height = base_height

            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "output.mp4")
            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30,
                (frame_width, frame_height)
            )

    while True:
        result = output_queue.get()
        if result is None: break

        original, infer, *rest = result
        infer = infer[0] if isinstance(infer, list) and len(infer) == 1 else infer

        if rest:
           result_cb=frame_with_detections = callback(original, infer, rest[0])
        else:
           result_cb=frame_with_detections = callback(original, infer)

        if isinstance(result_cb, tuple):
           frame_with_detections = result_cb[0]
        else:
           frame_with_detections = result_cb

        if fps_tracker is not None:
            fps_tracker.increment()

        if cap is not None:
            if save_stream_output:
                out.write(frame_with_detections)

            if not disable_display:
                cv2.imshow("Output", frame_with_detections)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if save_stream_output:
                        out.release()
                    cap.release()
                    cv2.destroyAllWindows()
                    break
        else:
            cv2.imwrite(os.path.join(output_dir, f"output_{image_id}.png"), frame_with_detections)

        image_id += 1

    if cap is not None:
        if save_stream_output and out is not None:
            out.release()
        if not disable_display:
            cv2.destroyAllWindows()
            cap.release()

    output_queue.task_done()
