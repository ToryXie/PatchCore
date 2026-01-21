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
    (WORKDIR / "models").mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"✅ Save Dir created: {save_path}")

    device = torch.device(f"cuda:{GPU_DEVICE}")
    LOGGER.info(f"✅ GPU set to: {GPU_DEVICE}")

    dataloaders: list[tuple[str, DataLoader]] = []
    patchcore_list: list[dict[str, str | list[str] | int | bool | Path]] = []
    for sub_dataset in args.datasets:
        model_path = WORKDIR / "models" / sub_dataset / args.model_name
        assert model_path.exists(), f"Model {args.model_name} does not exist: {model_path}"

        yaml_files = list(model_path.glob("*.yaml"))
        index_files = list(model_path.glob("*.index"))
        assert 0 < len(yaml_files) < 2, \
            f"Please check the number of yaml files: 0 < (Detected: {len(yaml_files)}) < 2."
        assert 0 < len(index_files) < 2, \
            f"Please check the number of index files: 0 < (Detected: {len(index_files)}) < 2."
        index_file = index_files[0]
        with open(yaml_files[0], 'r') as file:
            config_file = yaml.safe_load(file)

        backbone_name, backbone_parameters = list(config_file['backbone'].items())[0]
        backbone = backbones.load(backbone_name)
        assert backbone is not None, f"backbone is invalid or not exist: {backbone_name}."
        backbone_layers: list[str] = backbone_parameters["layers"]
        resize: int = backbone_parameters["resize"]
        image_size: int = backbone_parameters["image_size"]
        use_ivf = config_file["use_ivf"]["enable"]
        nprobe_scale = config_file["use_ivf"]["nprobe_scale"]
        embed_dim: int = config_file["embed_dim"]
        patch_size: int = config_file["patch_size"]
        fp16: bool = config_file["fp16"]

        patchcore_list.append({
            "name": backbone_name,
            "layers": backbone_layers,
            "resize": resize,
            "image_size": image_size,
            "use_ivf": use_ivf,
            "nprobe_scale": nprobe_scale,
            "embed_dim": embed_dim,
            "patch_size": patch_size,
            "fp16": fp16,
            "index_path": index_file
        })

        dataset = datasets.PatchCoreDataset("test", sub_dataset, resize, image_size)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        dataloaders.append((sub_dataset, dataloader))
    LOGGER.info("✅ Dataloaders & Model created.")

    for i, (dataset_name, dataloader) in enumerate(dataloaders):
        LOGGER.info(f"Current Dataset: {dataset_name.upper()} ({i + 1}/{len(dataloaders)})")

        patchcore_name = patchcore_list[i]["name"]
        LOGGER.info(f"Current model: {patchcore_name}")

        backbone = backbones.load(patchcore_name)
        patchcore = core.PatchCore(
            device,
            backbone,
            patchcore_list[i]["layers"],
            patchcore_list[i]["use_ivf"],
            patchcore_list[i]["nprobe_scale"],
            patchcore_list[i]["embed_dim"],
            patchcore_list[i]["resize"],
            patchcore_list[i]["image_size"],
            sampler.ApproximateGreedyCoresetSampler(),
            patchcore_list[i]["patch_size"],
            patchcore_list[i]["fp16"]
        )
        patchcore.load(patchcore_list[i]["index_path"])

        scores_lst, anomaly_type_lst, images_path_lst, masks_lst, masks_gt, time_lst = \
            patchcore.predict(dataloader)

        # Statistics
        results: list[dict[str, str | float]] = []
        for idx, mask in tqdm(enumerate(masks_lst), desc="Evaluating...", total=len(masks_lst)):
            if anomaly_type_lst[idx] == "good" or masks_gt[idx].max() == 0.:
                continue

            results.append(metrics.evaluate(anomaly_type_lst[idx], mask, masks_gt[idx]))

        metrics_save_path = save_path / dataset_name / patchcore_name
        metrics_save_path.mkdir(parents=True, exist_ok=True)
        utils.save_metrics(metrics_save_path, results, time_lst)
        if len(results) > 0:
            utils.plot_roc_curves(metrics_save_path, dataset_name, patchcore_name, results)
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

    LOGGER.info("✅ Evaluating run completed.")


if __name__ == "__main__":
    main()
