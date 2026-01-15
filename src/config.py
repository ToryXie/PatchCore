from pathlib import Path

import yaml

WORKDIR = Path(__file__).parent

with open(WORKDIR / "config.yaml", 'r') as file:
    config = yaml.safe_load(file)

BACKBONES: dict[str, list[str]] = config["model"]["backbones"]
BATCH_SIZE: int = config["model"]["batch_size"]
RESIZE: int = config["model"]["resize"]
IMAGE_SIZE: int = config["model"]["image_size"]
FP16: bool = config["model"]["fp16"]

USE_IVF: bool = config["model"]["use_ivf"]["enable"]
NPROBE_SCALE: int = config["model"]["use_ivf"]["nprobe_scale"]
EMBED_DIM: int = config["model"]["embed_dim"]
PATCH_SIZE: int = config["model"]["patch_size"]
GPU_DEVICE: int = config["model"]["gpu"]
NUM_WORKERS: int = config["model"]["num_workers"]

RANDOM_SEED: int = config["model"]["random_seed"]

DATASETS: list[str] = config["datasets"]
