import streamlit as st
import os

# ==================================================
# SIDEBAR
# ==================================================
def sidebar_layout():
    """
    Sidebar UI for user controls.
    """

    st.sidebar.header("⚙️ Settings")

    uploaded_video = st.sidebar.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    enable_tracking = st.sidebar.checkbox(
        "Enable tracking",
        value=True
    )

    enable_depth = st.sidebar.checkbox(
        "Enable depth estimation (MiDaS)",
        value=False
    )

    show_fps = st.sidebar.checkbox(
        "Show FPS",
        value=True
    )

    st.sidebar.markdown("---")

    run_button = st.sidebar.button("Run inference")

    return (
        uploaded_video,
        enable_tracking,
        enable_depth,
        show_fps,
        run_button,
    )


# ==================================================
# MAIN VIEW
# ==================================================
def main_layout(result, show_fps, inference_stats=None):
    """
    Main page layout showing inference results.
    """

    st.title("Video Inference")

    if not result:
        st.info("Please upload a video and run inference.")
        return

    tabs = st.tabs([
        "YOLO Result",
        "Depth (MiDaS)"
    ])

    # --------------------------------------------------
    # YOLO TAB
    # --------------------------------------------------
    with tabs[0]:
        st.subheader("YOLO Detection Result")

        yolo_path = result.get("yolo_video")
        if yolo_path:
            # IMPORTANT:
            # Read video as BYTES to avoid Streamlit caching issues
            with open(yolo_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes, format="video/mp4")
        else:
            st.warning("YOLO video not available.")

        if show_fps and inference_stats:
            st.markdown(
                f"""
                **Frames:** {inference_stats['frames']}  
                **Average FPS:** {inference_stats['fps']:.2f}  
                **Total time:** {inference_stats['time']:.2f} seconds
                """
            )

        if yolo_path:
            with open(yolo_path, "rb") as f:
                st.download_button(
                    label="Download",
                    data=f,
                    file_name="yolo_result.mp4",
                    mime="video/mp4",
                )

    # --------------------------------------------------
    # DEPTH TAB
    # --------------------------------------------------
    with tabs[1]:
        st.subheader("Depth Estimation (MiDaS)")

        depth_path = result.get("depth_video")

        if depth_path and os.path.exists(depth_path):
            with open(depth_path, "rb") as f:
                depth_bytes = f.read()

            st.video(depth_bytes, format="video/mp4")

            with open(depth_path, "rb") as f:
                st.download_button(
                    label="Download",
                    data=f,
                    file_name="depth_result.mp4",
                    mime="video/mp4",
                )
        else:
            st.info("Depth estimation was not enabled for this run.")

