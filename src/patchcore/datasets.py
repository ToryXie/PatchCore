from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

WORKDIR = Path(__file__).resolve().parents[2]


class PatchCoreDataset(Dataset):
    """Dataset class for PatchCore, handling image loading and preprocessing."""

    def __init__(self, split: str, model_name: str, resize: int = 256, image_size: int = 224):
        super().__init__()

        self.split = split
        self.model_name = model_name

        self.datasets_dir = WORKDIR / "datasets"
        self.image_data = self._get_image_data()

        self.transforms = nn.Sequential(
            v2.Resize(resize),
            v2.CenterCrop(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    def __getitem__(self, idx):
        data_entry = self.image_data[idx]
        image_path = data_entry["image_path"]
        image = decode_image(image_path, ImageReadMode.RGB)
        mask_gt = torch.zeros(1, image.shape[1], image.shape[2], dtype=torch.uint8)
        image = self.transforms(image)

        if self.split == "test":
            mask_gt_path = data_entry["mask_gt_path"]
            if mask_gt_path is not None:
                mask_gt = decode_image(mask_gt_path, ImageReadMode.GRAY)

        return {
            "image": image,
            "image_path": image_path,
            "anomaly_type": data_entry["anomaly_type"],
            "mask_gt": mask_gt
        }

    def __len__(self):
        return len(self.image_data)

    def _get_image_data(self) -> list[dict]:
        """Load and organize the image data from the dataset directory."""
        from train import LOGGER

        model_path = self.datasets_dir / self.model_name

        assert model_path.exists(), f"Images loading failed! Model does not exist: {self.model_name}."

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_data: list[dict] = []

        if self.split == "train":
            train_path = model_path / "train"
            assert train_path.exists(), f"Train folder does not exist: {train_path}."

            for file in sorted(train_path.iterdir()):
                if file.suffix.lower() in image_extensions:
                    image_data.append({
                        "image_path": str(file),
                        "mask_gt_path": None,
                        "anomaly_type": "train"
                    })

            assert image_data, f"Images loading failed! Model {self.model_name} contains no images!"

        elif self.split == "test":
            test_path = model_path / "test"

            assert test_path.exists(), f"Test folder does not exist: {test_path}."

            for anomaly_dir in sorted(test_path.iterdir()):
                if not anomaly_dir.is_dir():
                    continue

                anomaly_type = anomaly_dir.name

                if anomaly_type == "good":
                    for file in sorted(anomaly_dir.iterdir()):
                        if file.suffix.lower() in image_extensions:
                            image_data.append({
                                "image_path": str(file),
                                "mask_gt_path": None,
                                "anomaly_type": anomaly_type
                            })
                else:
                    img_dir = anomaly_dir / "img"
                    mask_gt_dir = anomaly_dir / "mask"

                    assert img_dir.exists(), f"Anomaly img folder does not exist: {img_dir}."
                    if not mask_gt_dir.exists():
                        LOGGER.warning(f"Anomaly mask folder does not exist: {anomaly_dir}.")

                    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in image_extensions])
                    for img_file in img_files:
                        mask_gt_file = mask_gt_dir / f"{img_file.stem}_mask{img_file.suffix}"
                        if not mask_gt_file.exists():
                            mask_gt_file = None

                        image_data.append({
                            "image_path": str(img_file),
                            "mask_gt_path": str(mask_gt_file) if mask_gt_file else None,
                            "anomaly_type": anomaly_type
                        })


        else:
            assert False, f"Split {self.split} not supported!"

        return image_data
