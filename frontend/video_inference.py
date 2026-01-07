import tempfile
import os
import sys

sys.path.append(os.path.abspath("../backend"))

from run_inference import run_inference


def run_inference_from_upload(
    uploaded_file,
    hef_path: str,
    labels_path: str,
    config_path: str,
    enable_tracking: bool = True,
    enable_depth: bool = False,
):
    """
    Streamlit upload -> temp video -> backend full-video inference
    """

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(uploaded_file.read())
        input_video_path = tmp_in.name

    try:
        result = run_inference(
            input_video_path=input_video_path,
            hef_path=hef_path,
            labels_path=labels_path,
            config_path=config_path,
            enable_tracking=enable_tracking,
            enable_depth=enable_depth,
        )
    finally:
        if os.path.exists(input_video_path):
            os.unlink(input_video_path)

    return result

