import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from config import GPU_DEVICE
from patchcore import backbones, core, datasets, metrics, sampler, utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

LOGGER = logging.getLogger(__name__)
WORKDIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description='PatchCore Model Evaluation')
    parser.add_argument('-m', '--model_name', type=str, required=True, help="Model name to evaluate.")
    parser.add_argument('-d', '--datasets', type=str, nargs='+', required=True,
                        help="One or more datasets to evaluate.")

    return parser.parse_args()


def main():
    args = parse_args()

    runs_dir = WORKDIR / "runs" / "evaluate"
    runs_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = WORKDIR / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_path = utils.increment_path(runs_dir / "exp")
    LOGGER.info(f"✅ Save Dir created: {save_path}")

    (WORKDIR / "models").mkdir(parents=True, exist_ok=True)
    model_path = WORKDIR / "models" / args.model_name
    assert model_path.exists(), f"Model {args.model_name} does not exist: {model_path}"

    device = torch.device(f"cuda:{GPU_DEVICE}")
    LOGGER.info(f"✅ GPU set to: {GPU_DEVICE}")

    yaml_files = list(model_path.glob("*.yaml"))
    assert 0 < len(yaml_files) < 2, f"Please check the number of yaml files: 0 < (Detected: {len(yaml_files)}) < 2."
    with open(yaml_files[0], 'r') as file:
        config_file = yaml.safe_load(file)

    BACKBONE_NAME, LAYERS = list(config_file['backbone'].items())[0]
    USE_IVF = config_file["use_ivf"]["enable"]
    NPROBE_SCALE = config_file["use_ivf"]["nprobe_scale"]
    EMBED_DIM: int = config_file["embed_dim"]
    RESIZE: int = config_file["resize"]
    IMAGE_SIZE: int = config_file["image_size"]
    PATCH_SIZE: int = config_file["patch_size"]
    FP16: bool = config_file["fp16"]
    LOGGER.info(f"✅ Config file loaded.")

    dataloaders: list[tuple[str, DataLoader]] = []
    for sub_dataset in args.datasets:
        dataset = datasets.PatchCoreDataset("test", sub_dataset, RESIZE, IMAGE_SIZE)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        dataloaders.append((sub_dataset, dataloader))
    LOGGER.info("✅ Dataloaders created.")

    LOGGER.info("Loading models...")
    backbone = backbones.load(BACKBONE_NAME)
    assert backbone is not None, f"backbone is invalid or not exist: {BACKBONE_NAME}."
    patchcore_instance = core.PatchCore(
        device,
        backbone,
        LAYERS,
        USE_IVF,
        NPROBE_SCALE,
        EMBED_DIM,
        RESIZE,
        IMAGE_SIZE,
        sampler.ApproximateGreedyCoresetSampler(),
        PATCH_SIZE,
        FP16
    )
    patchcore_instance.load(model_path)
    LOGGER.info("✅ Model loaded.")

    for i, (dataset_name, dataloader) in enumerate(dataloaders):
        LOGGER.info(f"Current Dataset: {dataset_name.upper()} ({i + 1}/{len(dataloaders)})")

        scores_lst, anomaly_type_lst, images_path_lst, masks_lst, masks_gt, time_lst = \
            patchcore_instance.predict(dataloader)

        # Statistics
        results: list[dict[str, str | float]] = []
        for idx, mask in tqdm(enumerate(masks_lst), desc="Evaluating...", total=len(masks_lst)):
            if anomaly_type_lst[idx] == "good" or masks_gt[idx].max() == 0.:
                continue

            results.append(metrics.evaluate(anomaly_type_lst[idx], mask, masks_gt[idx]))

        metrics_save_path = save_path / BACKBONE_NAME / f"{dataset_name}_IM{IMAGE_SIZE}"
        metrics_save_path.mkdir(parents=True, exist_ok=True)
        if len(results) > 0:
            utils.save_metrics(metrics_save_path, results, time_lst)
            utils.plot_roc_curves(metrics_save_path, dataset_name, BACKBONE_NAME, results)

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

    LOGGER.info("✅ Evaluating run completed.")


if __name__ == "__main__":
    main()
