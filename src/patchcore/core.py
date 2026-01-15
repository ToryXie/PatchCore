from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from . import common, db


class PatchCore(nn.Module):
    """Main PatchCore module."""

    def __init__(self,
                 device: torch.device,
                 backbone,
                 layers: list[str],
                 use_ivf: bool,
                 nprobe_scale: int,
                 embed_dim: int,
                 resize: int,
                 image_size: int,
                 feature_sampler,
                 patch_size: int,
                 use_half: bool):
        super().__init__()

        self.device = device
        self.backbone = backbone
        self.layers = layers
        self.use_ivf = use_ivf
        self.nprobe_scale = nprobe_scale
        self.embed_dim = embed_dim
        self.resize = resize
        self.image_size = image_size
        self.feature_sampler = feature_sampler
        self.patch_size = patch_size
        self.use_half = use_half

        self.feature_extractor = common.FeatureExtractor(self.backbone, self.layers, self.device)
        self.feature_extractor_half = common.FeatureExtractor(self.backbone, self.layers, self.device, True)
        self.preprocessing = common.Preprocessing()
        self.aggregator = common.Aggregator(self.embed_dim)
        self.patch_maker = PatchMaker(self.patch_size)
        self.anomaly_scorer = db.NearestNeighbourScorer(self.use_ivf, self.nprobe_scale, self.device.index)
        self.anomaly_segmentor = common.RescaleSegmentor(self.resize, self.image_size)

    def fit(self, dataloader: DataLoader):
        """Train the PatchCore model by building the memory bank with given data."""
        with torch.inference_mode():
            self._fill_memory_bank(dataloader)

    def predict(self, dataloader: DataLoader) \
            -> tuple[list[float], list[str], list[str], list[np.ndarray], list[np.ndarray], list[dict[str, float]]]:
        """Predict anomalies on the given dataset using the built memory bank."""
        with torch.inference_mode():
            return self._search_memory_bank(dataloader)

    def save(self, save_path: Path, name: str) -> None:
        """Save the trained model and its configuration."""
        from train import LOGGER

        LOGGER.info(f"Saving {name}...")
        self.anomaly_scorer.save(save_path, name, save_features=True)

        _, patchcore_name = name.split("-")
        save_data = {
            "backbone": {
                patchcore_name: self.layers
            },
            "resize": self.resize,
            "image_size": self.image_size,
            "fp16": self.use_half,
            "use_ivf": {
                "enable": self.use_ivf,
                "nprobe_scale": self.nprobe_scale
            },
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size
        }
        with open(save_path / f"{name}.yaml", 'w', encoding='utf-8') as file:
            yaml.dump(save_data, file, allow_unicode=True)

        LOGGER.info(f"✅ Save model & config success: {name}.")

    def load(self, model_path: Path) -> None:
        """Load the saved model from the specified path."""
        for file in model_path.iterdir():
            if file.is_file() and file.suffix == ".index":
                self.anomaly_scorer.load(file)

    def _fill_memory_bank(self, dataloader: DataLoader) -> None:
        """Construct the memory bank from the features extracted from the dataset."""
        all_features: list = []
        for im in tqdm(dataloader, desc="Computing features..."):
            images = im["image"].to(self.device)
            features, _ = self._extract_features(images)
            all_features.append(features.cpu())

        all_features: torch.Tensor = torch.cat(all_features, dim=0).to(self.device)
        sampled_features = self.feature_sampler.run(all_features)
        self.anomaly_scorer.fit(sampled_features)

    def _search_memory_bank(self, dataloader: DataLoader) \
            -> tuple[list[float], list[str], list[str], list[np.ndarray], list[np.ndarray], list[dict[str, float]]]:
        """Search the memory bank to predict anomalies for the given dataset."""
        scores_lst: list[float] = []
        anomaly_type_lst: list[str] = []
        images_path_lst: list[str] = []
        masks_lst: list[np.ndarray] = []
        masks_gt: list[np.ndarray] = []
        time_lst: list[dict[str, float]] = []
        for im in tqdm(dataloader, desc="Inferring..."):
            torch.cuda.synchronize()
            time_dict: dict[str, float] = {}
            start_time = time.perf_counter()

            images = im["image"].to(self.device)
            if self.use_half:
                images = images.half()
            images_path_lst.extend(im["image_path"])
            anomaly_type_lst.extend(im["anomaly_type"])
            batch_tensors = list(torch.split(im["mask_gt"], 1, dim=0))
            masks_gt.extend([tensor.squeeze().numpy() for tensor in batch_tensors])

            features, patch_shapes = self._extract_features(images, self.use_half)

            torch.cuda.synchronize()
            time_dict["backbone_time"] = (time.perf_counter() - start_time) * 1000
            start_time2 = time.perf_counter()

            anomaly_scores = self.anomaly_scorer.predict(features)

            torch.cuda.synchronize()
            time_dict["faiss_time"] = (time.perf_counter() - start_time2) * 1000

            anomaly_scores = self.patch_maker.unpatch_scores(anomaly_scores, images.shape[0])
            image_scores = self.patch_maker.score(anomaly_scores.unsqueeze(2))
            patch_scores = anomaly_scores.reshape(images.shape[0], patch_shapes[0][0], patch_shapes[0][1])
            masks = self.anomaly_segmentor.convert_to_mask(patch_scores)

            torch.cuda.synchronize()
            time_dict["Predict_time"] = (time.perf_counter() - start_time) * 1000

            time_lst.append(time_dict)

            for i in range(images.shape[0]):
                masks_lst.append(masks[i])
                scores_lst.append(image_scores[i].item())

        return scores_lst, anomaly_type_lst, images_path_lst, masks_lst, masks_gt, time_lst

    def _extract_features(self,
                          images: torch.Tensor,
                          use_half: bool = False) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Extract features from the backbone model at specified layers."""
        if use_half:
            features: list[torch.Tensor] = self.feature_extractor_half(images)
        else:
            features: list[torch.Tensor] = self.feature_extractor(images)
        patches_list: list[tuple[torch.Tensor, tuple[int, int]]] = []

        for feat in features:
            patches: tuple[torch.Tensor, tuple[int, int]] = self.patch_maker.patchify(feat)
            patches_list.append(patches)

        shapes: list[tuple[int, int]] = [i[1] for i in patches_list]
        preprocessed: torch.Tensor = self.preprocessing(patches_list)
        aggregated: torch.Tensor = self.aggregator(preprocessed)

        return aggregated, shapes


class PatchMaker:
    """Divide feature maps into patches."""

    def __init__(self, patch_size, stride=1):
        self.patch_size = patch_size
        self.stride = stride
        self.padding = (patch_size - 1) // 2

    def patchify(self, features: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Split the feature map into overlapping patches of size patch_size x patch_size."""
        B, C, H, W = features.shape

        # Use the sliding window to divide the image
        # into overlapping patches of size patch_size x patch_size
        # and flatten each patch.
        patches = F.unfold(
            features,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.padding
        )  # [B, C, H * W]

        patches = patches.view(B, C, self.patch_size, self.patch_size, -1)
        patches = patches.permute(0, 4, 1, 2, 3)  # [B, PATCHES_NUM, C, H, W]

        # Calculate the number of patches in h and w dimensions
        patch_h = (H + 2 * self.padding - self.patch_size) // self.stride + 1
        patch_w = (W + 2 * self.padding - self.patch_size) // self.stride + 1

        return patches, (patch_h, patch_w)

    @staticmethod
    def unpatch_scores(scores: torch.Tensor, batchsize: int) -> torch.Tensor:
        """Reorganize flattened patch scores into (batch, d) format."""
        num_patches_per_image = scores.shape[0] // batchsize

        return scores.reshape(batchsize, num_patches_per_image)

    @staticmethod
    def score(x: torch.Tensor) -> torch.Tensor:
        """Aggregate patch scores into a single image score."""
        while x.ndim > 1:
            x = x.max(dim=-1).values

        return x
