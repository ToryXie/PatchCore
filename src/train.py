import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (BACKBONES, BATCH_SIZE, DATASETS, EMBED_DIM, FP16, GPU_DEVICE, IMAGE_SIZE,
                    NPROBE_SCALE, NUM_WORKERS, PATCH_SIZE, RANDOM_SEED, RESIZE, USE_IVF)
from patchcore import backbones, core, datasets, metrics, sampler, utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

LOGGER = logging.getLogger(__name__)
WORKDIR = Path(__file__).resolve().parents[1]


def main():
    runs_dir = WORKDIR / "runs" / "train"
    runs_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = WORKDIR / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_path = utils.increment_path(runs_dir / "exp")
    LOGGER.info(f"✅ Save Dir created: {save_path}")

    device = torch.device(f"cuda:{GPU_DEVICE}")
    LOGGER.info(f"✅ GPU set to: {GPU_DEVICE}")

    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    LOGGER.info(f"✅ Random seed set to: {RANDOM_SEED}")

    dataloaders: list[tuple[str, dict[str, DataLoader]]] = []
    for sub_dataset in DATASETS:
        training_dataset = datasets.PatchCoreDataset("train", sub_dataset, RESIZE, IMAGE_SIZE)
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        testing_dataset = datasets.PatchCoreDataset("test", sub_dataset, RESIZE, IMAGE_SIZE)
        testing_dataloader = DataLoader(
            testing_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        dataloaders.append((sub_dataset, {"training": training_dataloader, "testing": testing_dataloader}))
    LOGGER.info("✅ Dataloaders created.")

    LOGGER.info("Loading backbone...")
    patchcore_list: list[tuple[str, core.PatchCore]] = []
    for backbone_name, backbone_layers in BACKBONES.items():
        backbone = backbones.load(backbone_name)
        assert backbone is not None, f"backbone is invalid or not exist: {backbone_name}."

        patchcore_instance = core.PatchCore(
            device,
            backbone,
            backbone_layers,
            USE_IVF,
            NPROBE_SCALE,
            EMBED_DIM,
            RESIZE,
            IMAGE_SIZE,
            sampler.ApproximateGreedyCoresetSampler(),
            PATCH_SIZE,
            FP16
        )

        patchcore_list.append((backbone_name, patchcore_instance))
    LOGGER.info("✅ Backbone loaded.")

    for dataset_name, dataloader in dataloaders:
        LOGGER.info(f"Current Dataset: {dataset_name.upper()}")

        for i, (patchcore_name, patchcore) in enumerate(patchcore_list):
            LOGGER.info(f"Training models: {patchcore_name} ({i + 1}/{len(patchcore_list)})")
            patchcore.fit(dataloader["training"])

            LOGGER.info(f"✅ {patchcore_name} training done.")

            patchcore_save_path = save_path / "models" / f"{dataset_name}_IM{IMAGE_SIZE}" / patchcore_name
            patchcore_save_path.mkdir(parents=True, exist_ok=True)

            patchcore.save(patchcore_save_path, f"{dataset_name}-{patchcore_name}")

        for i, (patchcore_name, patchcore) in enumerate(patchcore_list):
            LOGGER.info(f"Testing models: {patchcore_name} ({i + 1}/{len(patchcore_list)})")
            scores_lst, anomaly_type_lst, images_path_lst, masks_lst, masks_gt, time_lst = \
                patchcore.predict(dataloader["testing"])

            # Statistics
            results: list[dict[str, str | float]] = []
            for idx, mask in tqdm(enumerate(masks_lst), desc="Evaluating...", total=len(masks_lst)):
                if anomaly_type_lst[idx] == "good" or masks_gt[idx].max() == 0.:
                    continue

                results.append(metrics.evaluate(anomaly_type_lst[idx], mask, masks_gt[idx]))

            metrics_save_path = save_path / "results" / f"{dataset_name}_IM{IMAGE_SIZE}" / patchcore_name
            metrics_save_path.mkdir(parents=True, exist_ok=True)
            if len(results) > 0:
                utils.save_metrics(metrics_save_path, results, time_lst)
                utils.plot_roc_curves(metrics_save_path, dataset_name, patchcore_name, results)

            # Plotting
            for idx, anomaly_type in tqdm(enumerate(anomaly_type_lst), desc="Plotting...", total=len(anomaly_type_lst)):
                result_save_path = metrics_save_path / anomaly_type
                result_save_path.mkdir(parents=True, exist_ok=True)

                utils.plot_heatmap(
                    result_save_path,
                    scores_lst[idx],
                    images_path_lst[idx],
                    masks_lst[idx],
                    masks_gt[idx]
                )

    LOGGER.info("✅ Training run completed.")


if __name__ == "__main__":
    main()
