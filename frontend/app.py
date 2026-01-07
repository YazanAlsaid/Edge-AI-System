import streamlit as st
import os
import time

from layout import sidebar_layout, main_layout
from video_inference import run_inference_from_upload

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = "../backend/models/yolo11s_kitti_quant.hef"
LABELS_PATH = "../backend/configs/kitti_labels.txt"
CONFIG_PATH = "../backend/configs/config.json"

st.set_page_config(layout="wide")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None

if "inference_stats" not in st.session_state:
    st.session_state.inference_stats = None

if "run_id" not in st.session_state:
    st.session_state.run_id = 0

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
uploaded_video, enable_tracking, enable_depth, show_fps, run= sidebar_layout()

# --------------------------------------------------
# RUN INFERENCE
# --------------------------------------------------
if uploaded_video and run:
    st.session_state.run_id += 1
    start_time = time.time()

    with st.spinner("Inference is running, please wait..."):
        result = run_inference_from_upload(
            uploaded_file=uploaded_video,
            hef_path=MODEL_PATH,
            labels_path=LABELS_PATH,
            config_path=CONFIG_PATH,
            enable_tracking=enable_tracking,
            enable_depth=enable_depth,
        )

    total_time = time.time() - start_time

    st.session_state.result = result
    st.session_state.inference_stats = {
        "frames": result["frames"],
        "fps": result["frames"] / total_time if total_time > 0 else 0,
        "time": total_time,
    }

    st.success("Inference completed")

# --------------------------------------------------
# MAIN VIEW
# --------------------------------------------------
if st.session_state.result:
    main_layout(
        result=st.session_state.result,
        show_fps=show_fps,
        inference_stats=st.session_state.inference_stats,
    )
