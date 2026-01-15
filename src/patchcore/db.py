from pathlib import Path
import pickle

import faiss
import torch


class FaissNN:
    """Faiss nearest neighbor search."""

    def __init__(self, use_ivf: bool, nprobe_scale: int, gpu_device: int):
        self.use_ivf = use_ivf
        self.nprobe_scale = nprobe_scale
        self.gpu_device = gpu_device

        self.search_index = None
        self.dimension = None

    def fit(self, features: torch.Tensor) -> None:
        """Build the index using the provided features."""
        self.dimension = features.shape[-1]

        if self.use_ivf:
            gpu_config = faiss.GpuIndexIVFFlatConfig()
            gpu_config.device = self.gpu_device
            self.search_index = faiss.GpuIndexIVFFlat(
                faiss.StandardGpuResources(),
                self.dimension,
                features.shape[0] // 39,
                faiss.METRIC_L2,
                gpu_config
            )
            if not self.search_index.is_trained:
                self.search_index.train(features.cpu().numpy())  # type: ignore
        else:
            gpu_config = faiss.GpuIndexFlatConfig()
            gpu_config.device = self.gpu_device
            self.search_index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), self.dimension, gpu_config)
        self.search_index.add(features.cpu().numpy())  # type: ignore

    def search(self, query_features: torch.Tensor, n_nearest_neighbours: int) -> torch.Tensor:
        """Perform the search to find nearest neighbors."""
        if self.use_ivf:
            self.search_index.nprobe = self.search_index.ntotal // 39 // self.nprobe_scale
            distances, indices = self.search_index.search(query_features.cpu().numpy(), n_nearest_neighbours)
        else:
            distances, indices = self.search_index.search(query_features.cpu().numpy(), n_nearest_neighbours)

        return torch.from_numpy(distances)

    def save(self, index_path: str) -> None:
        """Save the index to a file."""
        faiss.write_index(faiss.index_gpu_to_cpu(self.search_index), index_path)

    def load(self, index_path: str) -> None:
        """Load the index from a file onto the specified GPU."""
        self.search_index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            self.gpu_device,
            faiss.read_index(index_path)
        )


class NearestNeighbourScorer:
    """Uses nearest neighbor search to compute anomaly scores based on feature distances."""

    def __init__(self, use_ivf: bool, nprobe_scale: int, gpu_device: int, n_nearest_neighbours: int = 1):
        self.n_nearest_neighbours = n_nearest_neighbours

        self.nn_method = FaissNN(use_ivf, nprobe_scale, gpu_device)
        self.detection_features: torch.Tensor | None = None

    def fit(self, detection_features: torch.Tensor) -> None:
        """Build the feature index."""
        from train import LOGGER

        self.detection_features = detection_features
        self.nn_method.fit(self.detection_features)
        LOGGER.info(f"✅ Build features success. Numbers of index: {self.nn_method.search_index.ntotal}.")

    def predict(self, query_features: torch.Tensor) -> torch.Tensor:
        """Search the features for nearest neighbors."""
        query_distances = self.nn_method.search(query_features, self.n_nearest_neighbours)
        anomaly_scores = torch.mean(query_distances, dim=-1)

        return anomaly_scores

    def save(self,
             save_folder: Path,
             name: str,
             save_features: bool = False) -> None:
        """Save the index and feature data."""
        self.nn_method.save(self._index_file(save_folder, name))
        if save_features:
            self._save_features(self._features_file(save_folder, name), self.detection_features)

    def load(self, file: Path) -> None:
        """Load the index from the specified file."""
        self.nn_method.load(str(file))

    @staticmethod
    def _save_features(file_path: str, features: torch.Tensor) -> None:
        """Save the feature data to a file."""
        with open(file_path, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)  # type: ignore

    @staticmethod
    def _features_file(folder: Path, name: str) -> str:
        """Generate the feature file name."""
        return str(folder / f"{name}.pkl")

    @staticmethod
    def _index_file(folder: Path, name: str) -> str:
        """Generate the index file name."""
        return str(folder / f"{name}.index")
