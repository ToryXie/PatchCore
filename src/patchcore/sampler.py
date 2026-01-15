import torch
import torch.nn as nn
from tqdm import tqdm


class ApproximateGreedyCoresetSampler:
    """Approximate greedy coreset sampling algorithm."""

    def __init__(self, percentage: float = 0.1, starting_points_num: int = 10, downsample_dimension: int = 128):
        assert 0 < percentage < 1, "The sampling rate should be in the range (0, 1)."

        self.percentage = percentage
        self.downsample_dimension = downsample_dimension
        self.starting_points_num = starting_points_num

    def run(self, features: torch.Tensor) -> torch.Tensor:
        """Perform coreset sampling on the given features."""
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]

        return features

    def _reduce_features(self, features: torch.Tensor) -> torch.Tensor:
        """Downsample the feature set to a lower dimensionality."""
        if features.shape[1] == self.downsample_dimension:
            return features
        else:
            mapper = nn.Linear(features.shape[1], self.downsample_dimension, bias=False).to(features.device)

        return mapper(features)

    @staticmethod
    def _compute_distance(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        """Compute the Euclidean distance between two matrices."""
        a_square = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_square = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_square + b_square).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> list[int]:
        """Compute the indices of the selected coreset samples."""
        starting_points_num = min(self.starting_points_num, len(features))
        start_points = torch.randperm(len(features))[:starting_points_num].tolist()

        # Compute the Euclidean distance between all pairs of samples, resulting in an N x N symmetric matrix.
        distance_matrix = self._compute_distance(features, features[start_points])
        # Compute the L2 norm of the distance from each sample to all other samples.
        coreset_distances = torch.mean(distance_matrix, dim=-1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        # Greedy selection.
        for _ in tqdm(range(num_coreset_samples), desc="Subsampling..."):
            # Select the sample that is furthest from the current coreset.
            select_idx = torch.argmax(coreset_distances).item()
            coreset_indices.append(select_idx)

            # Compute the distance from the selected sample to all other points.
            coreset_select_distance = self._compute_distance(features, features[select_idx: select_idx + 1])

            # Update the distances: for each point, keep the minimum distance to any of the selected samples
            coreset_distances = torch.cat([coreset_distances.unsqueeze(1), coreset_select_distance], dim=-1)
            coreset_distances = torch.min(coreset_distances, dim=1).values

        return coreset_indices
