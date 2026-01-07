import torch
import cv2
import numpy as np


class MiDaSSmall:
    """
    MiDaS Small (fast, lightweight) depth estimator
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load MiDaS Small model
        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",
            pretrained=True
        ).to(self.device)

        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms"
        )
        self.transform = midas_transforms.small_transform

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Predict depth map for a single frame.
        Returns uint8 depth image.
        """

        # BGR -> RGB
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # Normalize to 0–255
        depth_norm = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        return depth_norm
