from pathlib import Path
from pathlib import Path
import shutil
import random
import sys

from config import KITTI_IMAGES, KITTI_LABELS, KITTI_YOLO_DIR

KITTI_CLASS_MAP = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7
}


class KITTI2YoloDataset:
    """
    Converts KITTI object detection dataset into YOLO format.
    - Converts bounding boxes to YOLO xywh normalized format.
    - Removes DontCare regions.
    - Performs train/val split.
    - Writes YOLO labels & images.
    - Generates data.yaml for Ultralytics YOLO.
    """

    def __init__(self,
        kitti_img_dir: Path = KITTI_IMAGES,
        kitti_label_dir: Path = KITTI_LABELS,
        output_dir: Path = KITTI_YOLO_DIR,
        val_split: float = 0.1):

        self.kitti_img_dir = Path(kitti_img_dir)
        self.kitti_label_dir = Path(kitti_label_dir)
        self.output_dir = Path(output_dir)
        self.val_split = val_split

        self.train_img_out = self.output_dir / "images/train"
        self.val_img_out   = self.output_dir / "images/val"
        self.train_lbl_out = self.output_dir / "labels/train"
        self.val_lbl_out   = self.output_dir / "labels/val"

        self._prepare_output_dirs()

    def _prepare_output_dirs(self):
        """Create output directories."""
        for d in [self.train_img_out, self.val_img_out,
            self.train_lbl_out, self.val_lbl_out]:
            d.mkdir(parents=True, exist_ok=True)

    def _parse_kitti_label(self, label_path: Path):
        """Parse KITTI TXT label file."""
        objects = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split(" ")

                cls = parts[0]
                if cls == "DontCare":
                    continue  # skip

                if cls not in KITTI_CLASS_MAP:
                    continue

                x1, y1, x2, y2 = map(float, parts[4:8])
                objects.append((cls, x1, y1, x2, y2))

        return objects
    def list_kitti_classes(self):
        """
        Scans all KITTI label files and returns a sorted list of all
        class names that appear in the dataset.
        """
        classes = set()

        for label_path in self.labels:
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) > 0:
                        classes.add(parts[0])  # class name is first entry

        return sorted(list(classes))

    def _convert_to_yolo(self, cls, x1, y1, x2, y2, img_w, img_h):
        """Convert KITTI bbox to YOLO normalized (xc, yc, w, h)."""
        xc = (x1 + x2) / 2.0 / img_w
        yc = (y1 + y2) / 2.0 / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        return KITTI_CLASS_MAP[cls], xc, yc, w, h

    def convert_image(self, img_path: Path, label_path: Path, out_img_dir: Path, out_lbl_dir: Path):
        """Convert one KITTI image and label to YOLO format."""

        # Copy image
        shutil.copy(img_path, out_img_dir / img_path.name)

        # Read image dimensions
        import cv2
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]

        # Parse labels
        objects = self._parse_kitti_label(label_path)

        # Write YOLO label file
        out_path = out_lbl_dir / (img_path.stem + ".txt")
        with open(out_path, "w") as f:
            for cls, x1, y1, x2, y2 in objects:
                cls_id, xc, yc, w, h = self._convert_to_yolo(cls, x1, y1, x2, y2, img_w, img_h)
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    def convert(self):
        """Convert full dataset (train + val split)."""

        print("Loading KITTI dataset...")

        # Load all image files
        all_images = sorted(list(self.kitti_img_dir.glob("*.png")))
        random.shuffle(all_images)

        val_count = int(len(all_images) * self.val_split)

        val_imgs = all_images[:val_count]
        train_imgs = all_images[val_count:]

        print(f"Train images: {len(train_imgs)}")
        print(f"Val images:   {len(val_imgs)}")

        for img_path in train_imgs:
            label_file = self.kitti_label_dir / (img_path.stem + ".txt")
            if label_file.exists():
                self.convert_image(img_path, label_file, self.train_img_out, self.train_lbl_out)

        for img_path in val_imgs:
            label_file = self.kitti_label_dir / (img_path.stem + ".txt")
            if label_file.exists():
                self.convert_image(img_path, label_file, self.val_img_out, self.val_lbl_out)

        print("Conversion complete.")

    def generate_yaml(self):
        """Generate YOLO data.yaml."""
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write("train: images/train\n")
            f.write("val: images/val\n\n")
            f.write(f"nc: {len(KITTI_CLASS_MAP)}\n")
            f.write(f"names: {list(KITTI_CLASS_MAP.keys())}\n")

        print(f"data.yaml created at {yaml_path}")


if __name__ == "__main__":
    
    converter = KITTI2YoloDataset()
    print(f"detected classes : {converter.list_kitti_classes()}")
    converter.convert()
    converter.generate_yaml()
