from pathlib import Path

"""
Configuration file for converting:
- KITTI → YOLO

This file defines all dataset paths and global settings.
"""

# Path to KITTI object detection labels (.txt files)
KITTI_LABELS = Path("/Users/yazanalsaid/Downloads/training/label_2")

# Path to KITTI object detection images (.png files)
KITTI_IMAGES = Path("/Users/yazanalsaid/Downloads/data_object_image_2/training/image_2")
