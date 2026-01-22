import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (BACKBONES, BATCH_SIZE, DATASETS, EMBED_DIM, FP16, GPU_DEVICE,
                    NPROBE_SCALE, NUM_WORKERS, PATCH_SIZE, RANDOM_SEED, USE_IVF)
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

    LOGGER.info("Loading backbone config...")
    patchcore_list: list[dict[str, str | list[str] | int]] = []
    for backbone_name, backbone_parameters in BACKBONES.items():
        backbone = backbones.load(backbone_name)
        assert backbone is not None, f"backbone is invalid or not exist: {backbone_name}."
        backbone_layers: list[str] = backbone_parameters["layers"]
        resize: int = backbone_parameters["resize"]
        image_size: int = backbone_parameters["image_size"]

        patchcore_list.append({
            "name": backbone_name,
            "layers": backbone_layers,
            "resize": resize,
            "image_size": image_size
        })
    LOGGER.info("✅ Backbone config loaded.")

    dataloaders: list[tuple[str, dict[str, DataLoader]]] = []
    for i, sub_dataset in enumerate(DATASETS):
        training_dataset = datasets.PatchCoreDataset(
            "train",
            sub_dataset,
            patchcore_list[i]["resize"],
            patchcore_list[i]["image_size"]
        )
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        testing_dataset = datasets.PatchCoreDataset(
            "test",
            sub_dataset,
            patchcore_list[i]["resize"],
            patchcore_list[i]["image_size"]
        )
        testing_dataloader = DataLoader(
            testing_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        dataloaders.append((sub_dataset, {"training": training_dataloader, "testing": testing_dataloader}))
    LOGGER.info("✅ Dataloaders created.")

    for dataset_name, dataloader in dataloaders:
        LOGGER.info(f"Current Dataset: {dataset_name.upper()}")

        for i, patchcore_dict in enumerate(patchcore_list):
            patchcore_name = patchcore_dict["name"]
            LOGGER.info(f"Current model: {patchcore_name} ({i + 1}/{len(patchcore_list)})")

            backbone_layers = patchcore_dict["layers"]
            resize = patchcore_dict["resize"]
            image_size = patchcore_dict["image_size"]
            backbone = backbones.load(patchcore_name)

            patchcore = core.PatchCore(
                device,
                backbone,
                backbone_layers,
                USE_IVF,
                NPROBE_SCALE,
                EMBED_DIM,
                resize,
                image_size,
                sampler.ApproximateGreedyCoresetSampler(),
                PATCH_SIZE,
                FP16
            )

            # Training
            LOGGER.info(f"Training model: {patchcore_name}")
            patchcore.fit(dataloader["training"])
            LOGGER.info(f"✅ {patchcore_name} training done.")

            patchcore_save_path = save_path / "models" / dataset_name / patchcore_name
            patchcore_save_path.mkdir(parents=True, exist_ok=True)

            patchcore.save(patchcore_save_path, f"{dataset_name}-{patchcore_name}")

            # Testing
            LOGGER.info(f"Testing model: {patchcore_name}")
            scores_lst, anomaly_type_lst, images_path_lst, masks_lst, masks_gt, time_lst = \
                patchcore.predict(dataloader["testing"])

            # Statistics
            results: list[dict[str, str | float]] = []
            for idx, mask in tqdm(enumerate(masks_lst), desc="Evaluating...", total=len(masks_lst)):
                if anomaly_type_lst[idx] == "good" or masks_gt[idx].max() == 0.:
                    continue

                results.append(metrics.evaluate(anomaly_type_lst[idx], mask, masks_gt[idx]))

            metrics_save_path = save_path / "results" / dataset_name / patchcore_name
            metrics_save_path.mkdir(parents=True, exist_ok=True)
            utils.save_metrics(metrics_save_path, results, time_lst)
            if len(results) > 0:
                utils.plot_curves(metrics_save_path, dataset_name, patchcore_name, results)
            else:
                LOGGER.info("No masks detected, no roc metrics generated.")

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
