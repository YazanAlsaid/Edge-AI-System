import sys
import os
import cv2
import time
import queue
import threading
import numpy as np
from typing import Dict
from types import SimpleNamespace
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from common.hailo_inference import HailoInfer
from common.toolbox import (
    FrameRateTracker,
    get_labels,
    load_json_file,
)
from preprocessing.preprocess import preprocess_for_streamlit
from postprocessing.postprocess import inference_result_handler
from common.tracker.ByteTrack.byte_tracker import BYTETracker
from common.depth.miDaS_depth import MiDaSSmall


class inferencePipeline:
    """
    3-Thread inference pipeline:
    - Preprocess Thread
    - Inference Thread
    - Postprocess Thread
    """

    # --------------------------------------------------
    # Init
    # --------------------------------------------------
    def __init__(
        self,
        hef_path: str,
        labels_path: str,
        config_path: str,
        enable_tracking: bool = True,
    ):
        # Queues
        self.input_queue = queue.Queue()     # Main -> Preprocess
        self.infer_queue = queue.Queue()     # Preprocess -> Inference
        self.post_queue = queue.Queue()      # Inference -> Postprocess
        self.result_queue = queue.Queue()    # Postprocess -> Main

        self.stop_event = threading.Event()

        # Core
        self.enable_tracking = enable_tracking
        self.enable_depth = False
        self.depth_estimator = None

        # -------- Hailo --------
        self.hailo = HailoInfer(hef_path, batch_size=1)
        self.input_h, self.input_w, _ = self.hailo.get_input_shape()

        # -------- Config --------
        self.labels = get_labels(labels_path)
        self.config_data = load_json_file(config_path)
        # -------- Tracker --------
        self.tracker = None
        if enable_tracking:
            tracker_cfg = self.config_data.get("visualization_params", {}).get("tracker", {})
            self.tracker = BYTETracker(SimpleNamespace(**tracker_cfg))

        # -------- Performance --------
        self.fps_tracker = FrameRateTracker()
        self.frame_count = 0

        # -------- Threads --------
        self.preprocess_thread = threading.Thread(
            target=self._preprocess_loop, daemon=True
        )
        self.infer_thread = threading.Thread(
            target=self._infer_loop, daemon=True
        )
        self.postprocess_thread = threading.Thread(
            target=self._postprocess_loop, daemon=True
        )

        self.preprocess_thread.start()
        self.infer_thread.start()
        self.postprocess_thread.start()

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def enable_depth_estimation(self, enable: bool):
        self.enable_depth = enable
        if enable and self.depth_estimator is None:
            self.depth_estimator = MiDaSSmall()

    def reset(self):
        self.frame_count = 0
        self.fps_tracker = FrameRateTracker()
        if self.enable_tracking:
            tracker_cfg = self.config_data.get("visualization_params", {}).get("tracker", {})
            self.tracker = BYTETracker(SimpleNamespace(**tracker_cfg))

    def stop(self):
        self.stop_event.set()
        self.preprocess_thread.join()
        self.infer_thread.join()
        self.postprocess_thread.join()
        self.hailo.close()

    # --------------------------------------------------
    # MAIN ENTRY (Main Thread)
    # --------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Main-thread entry.
        Only pushes frame and waits for final result.
        """
        self.input_queue.put(frame)

        try:
            return self.result_queue.get(timeout=1.0)
        except queue.Empty:
            return {
                "image": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                "depth": None,
                "stats": {},
            }

    # --------------------------------------------------
    # THREAD 1: Preprocess
    # --------------------------------------------------
    def _preprocess_loop(self):
        while not self.stop_event.is_set():
            try:
                frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            processed = preprocess_for_streamlit(
                frame, self.input_w, self.input_h
            )

            self.infer_queue.put((frame, processed))

    # --------------------------------------------------
    # THREAD 2: Inference (TEST-SCRIPT STYLE)
    # --------------------------------------------------
    def _infer_loop(self):
        while not self.stop_event.is_set():
            try:
                frame, processed = self.infer_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            infer_result = self._run_inference(processed)

            depth_map = None
            if self.enable_depth and self.depth_estimator:
                depth_map = self.depth_estimator.predict(frame)

            self.post_queue.put((frame, infer_result, depth_map))

    # --------------------------------------------------
    # THREAD 3: Postprocess
    # --------------------------------------------------
    def _postprocess_loop(self):
        while not self.stop_event.is_set():
            try:
                frame, infer_result, depth_map = self.post_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            annotated_bgr = inference_result_handler(
                original_frame=frame,
                infer_results=infer_result,
                labels=self.labels,
                config_data=self.config_data,
                tracker=self.tracker,
                depth_map=depth_map,
            )

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            self.frame_count += 1
            self.fps_tracker.increment()

            stats = {
                "frame_number": self.frame_count,
                "fps": self.fps_tracker.fps,
                "timestamp": time.strftime("%H:%M:%S"),
            }

            self.result_queue.put({
                "image": annotated_rgb,
                "depth": depth_map,
                "stats": stats,
            })

    # --------------------------------------------------
    # INFERENCE CORE (UNCHANGED, TEST-SCRIPT STYLE)
    # --------------------------------------------------
    def _run_inference(self, processed_frame: np.ndarray):
        result_queue = queue.Queue()

        def callback(completion_info, bindings_list):
            if completion_info.exception:
                result_queue.put(("error", completion_info.exception))
                return

            bindings = bindings_list[0]
            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            else:
                result = {
                    name: np.expand_dims(
                        bindings.output(name).get_buffer(), axis=0
                    )
                    for name in bindings._output_names
                }

            result_queue.put(("ok", result))

        self.hailo.run([processed_frame], callback)

        status, payload = result_queue.get(timeout=30.0)
        if status == "error":
            raise RuntimeError(payload)

        return payload
