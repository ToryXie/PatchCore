import copy
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """Extract features from specified layers of the backbone model."""

    def __init__(self, backbone, layers: list[str], device: torch.device, use_half: bool = False):
        super().__init__()

        self.layers = layers
        self.device = device

        self.features = []
        self.hooks = []

        self.backbone = copy.deepcopy(backbone).to(self.device).eval()
        if use_half:
            self.backbone = self.backbone.half()

        last_layer = layers[-1]
        for layer_name in layers:
            layer = self._get_layer(self.backbone, layer_name)
            stop = (layer_name == last_layer)
            hook = layer.register_forward_hook(self._make_hook(stop))
            self.hooks.append(hook)

    def forward(self, x):
        self.features.clear()
        try:
            _ = self.backbone(x)
        except RuntimeError:
            pass

        return self.features

    @staticmethod
    def _get_layer(model, layer_name: str) -> nn.Sequential:
        """Retrieve the specified layer from the model."""
        parts = layer_name.split('.')
        layer = model
        for part in parts:
            layer = getattr(layer, part)

        return layer

    def _make_hook(self, stop=False) -> Callable:
        """Create a forward hook function to capture the output features of the specified layer."""

        def hook(module, input, output):
            self.features.append(output)

            if stop:
                raise RuntimeError("Reached last layer, stopping forward.")

        return hook

    def __del__(self):
        for hook in self.hooks:
            hook.remove()


class Preprocessing(nn.Module):
    """Align and unify features with different dimensions and spatial resolutions."""

    def __init__(self, output_dim: int = 1024):
        super().__init__()

        self.output_dim = output_dim

    def forward(self, patch_features: list[tuple[torch.Tensor, tuple[int, int]]]) -> torch.Tensor:
        aligned_features = self._align_patches(patch_features)
        mapped_features = self._map_dimensions(aligned_features)

        return mapped_features

    @staticmethod
    def _align_patches(features: list[tuple[torch.Tensor, tuple[int, int]]]) -> list[torch.Tensor]:
        """
        Align feature patches of different layers to the same spatial resolution.
        Use the first layer as the reference resolution and apply bilinear interpolation to align the other layers.
        """
        anchor_feat, (anchor_h, anchor_w) = features[0]
        aligned = [anchor_feat.reshape(-1, *anchor_feat.shape[2:])]

        for feat, (h, w) in features[1:]:
            B, _, C, H, W = feat.shape
            feat = feat.reshape(B, h, w, C, H, W).permute(0, 3, 4, 5, 1, 2)
            feat = feat.view(-1, h, w)
            feat = F.interpolate(
                feat.unsqueeze(1),
                size=(anchor_h, anchor_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(1)

            feat = feat.reshape(B, C, H, W, anchor_h, anchor_w).permute(0, 4, 5, 1, 2, 3)
            feat = feat.reshape(-1, C, H, W)

            aligned.append(feat)

        return aligned

    def _map_dimensions(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Unify the feature dimensions across different layers."""
        mapped = []

        for feat in features:
            feat = feat.reshape(feat.shape[0], -1)
            feat.unsqueeze_(1)  # [N, C] -> [N, 1, C]
            feat = F.adaptive_avg_pool1d(feat, self.output_dim)  # [N, 1, output_dim]
            feat.squeeze_(1)  # [N, output_dim]
            mapped.append(feat)

        # Stack the features from all layers
        return torch.stack(mapped, dim=1)  # [N, num_layers, output_dim]


class Aggregator(nn.Module):
    """Compress the dimensions of the input features."""

    def __init__(self, target_dim: int):
        super().__init__()

        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(features.shape[0], 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        features = features.reshape(features.shape[0], -1)

        return features


class RescaleSegmentor:
    """Convert patch anomaly scores into anomaly mask maps."""

    def __init__(self, original_size: int, target_size: int):
        self.original_size = original_size
        self.target_size = target_size

        self.smoothing = 4

    def convert_to_mask(self, patch_scores: torch.Tensor) -> list[np.ndarray]:
        patch_scores.unsqueeze_(1)
        patch_scores = F.interpolate(patch_scores, self.target_size, mode='bilinear', align_corners=False)
        patch_scores.squeeze_(1)
        patch_scores = patch_scores.numpy()

        smoothed_masks = []
        for patch_score in patch_scores:
            ksize = int(6 * self.smoothing + 1)
            if ksize % 2 == 0:
                ksize += 1  # Ensure the kernel size is odd

            smoothed = cv2.GaussianBlur(
                patch_score,
                ksize=(ksize, ksize),
                sigmaX=self.smoothing,
                sigmaY=self.smoothing,
                borderType=cv2.BORDER_REFLECT101
            )

            border = (self.original_size - self.target_size) // 2
            smoothed = cv2.copyMakeBorder(
                smoothed,
                border, border, border, border,
                cv2.BORDER_CONSTANT,
                value=0
            )

            smoothed_masks.append(smoothed)

        return smoothed_masks
