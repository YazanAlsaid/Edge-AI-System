# Edge AI Video Inference on Raspberry Pi 5 with Hailo-8L

This project performs **video inference** on a **Raspberry Pi 5** using the **Hailo-8L 13 TOPS AI accelerator** and the **HailoRT runtime**.  
Inference is executed on **pre-recorded video files** and is controlled via a **Streamlit web interface**.

> ⚠️ **Important**  
> This project **must be executed on a Raspberry Pi with Hailo hardware**.  
> It **will not run on macOS or standard PCs**, because it depends on the Hailo runtime and accelerator.

---

## Features

- Offline video inference on the edge  
- Optimized YOLO11s model compiled for **Hailo-8L**
- Streamlit-based web UI
- Modular backend / frontend architecture
- KITTI → YOLO11 dataset conversion tools
- Training & compilation notebooks included

---

## Project Structure

```text
edge-ai/
├── backend/
│   ├── run_inference.py          # Main entry point for video inference
│   ├── common/
│   │   ├── hailo_inference.py
│   │   ├── toolbox.py
│   │   ├── tracker/
│   │   │   └── ByteTrack/
│   │   │       ├── basetrack.py
│   │   │       ├── byte_tracker.py
│   │   │       ├── kalman_filter.py
│   │   │       └── matching.py
│   │   └── depth/
│   │       └── miDaS_depth.py
│   ├── preprocessing/
│   │   └── preprocess.py
│   ├── postprocessing/
│   │   ├── detection_utils.py
│   │   └── postprocess.py
│   ├── inference/
│   ├── evaluation/
│   ├── configs/
│   └── models/
│       └── yolo11s_kitti_quant.hef
│
├── frontend/
│   ├── app.py
│   ├── layout.py
│   └── video_inference.py
│
├── converter/
│   ├── kitti2yolo.py
│   └── config.py
│
├── data/
│   └── *.mp4
│
├── Notebooks/
│   ├── yolo11s_kitti_training.ipynb
│   └── Yolo11s-Hailo8l(.hef)-Compilation.ipynb
│
├── requirements.txt
└── README.md
```

---

## Hardware Requirements

- Raspberry Pi 5  
- Hailo-8L AI Accelerator  
- Raspberry Pi OS (64-bit)  
- Pre-recorded video files (`.mp4`)

---

## Software Requirements

- Python 3.10+
- HailoRT (installed on Raspberry Pi)
- pip
- virtualenv (recommended)
- Streamlit

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YazanAlsaid/Edge-AI-System.git
cd edge-ai

# 2. Create and activate virtual environment
python3 -m venv venv --system-site-packages # Give the virtual environment access to the system site-packages directory.
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

```
---

## How to run

1) Start the Streamlit application

```bash
streamlit run frontend/app.py
```

2) Open the Web Interface

```bash
# http://localhost:8501
``` 

From there you can:

- Upload a pre-recorded video file
- Enable or disable optional features:
  - Depth estimation (MiDaS)
  - Object tracking
  - show FPS
- Run inference
- View the results


<img width="1715" height="954" alt="Screenshot 2026-01-06 at 01 29 38" src="https://github.com/user-attachments/assets/d2f6ab53-10a8-44ca-b25d-0908338c2a47" />

## Video Infernece Example 

https://github.com/user-attachments/assets/09fbbb06-ca72-46c8-8bda-10f53628c474

## Model

```text
Model: YOLO11s
Dataset: KITTI (https://www.cvlibs.net/datasets/kitti/index.php)
Format: Hailo Executable File (.hef) Quantized and compiled for Hailo-8L
Model location: backend/models/yolo11s_kitti_quant.hef
```

## Dataset Conversion

To convert KITTI annotations to YOLO format, run:

```bash
python converter/kitti2yolo.py
# config file : converter/config.py
```

## Training & Compilation

```text
icluded notebooks:
1) YOLO training : Notebooks/yolo11s_kitti_training.ipynb

2) Hailo compilation (.hef) : Notebooks/Yolo11s-Hailo8l(.hef)-Compilation.ipynb

steps are typically executed on Google Colab, not on the Raspberry Pi.

```
## License
This project is intended for educational and research purposes.

## Author
- **Yazan Alsaid**
