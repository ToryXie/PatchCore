from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from train import LOGGER


def increment_path(base_path: Path) -> Path:
    """Generate a unique path by incrementing the base name."""
    base_path = Path(base_path)

    parent_dir = base_path.parent
    base_name = base_path.name

    existing_folders = []
    if parent_dir.exists():
        for item in parent_dir.iterdir():
            if item.is_dir() and item.name.startswith(base_name):
                existing_folders.append(item.name)

    max_num = -1
    for folder in existing_folders:
        suffix = folder[len(base_name):]
        num = int(suffix) if len(suffix) > 0 else 0
        max_num = max(max_num, num)

    new_num = max_num + 1
    new_name = f"{base_name}{new_num}" if new_num > 0 else base_name
    new_path = parent_dir / new_name

    new_path.mkdir(parents=True, exist_ok=True)

    return new_path


def save_metrics(csv_save_path: Path,
                 result: list[dict[str, float]],
                 time_lst: list[dict[str, float]]) -> None:
    """Save evaluation metrics to a CSV file."""
    # AUROC metrics
    if len(result) > 0:
        anomaly_dict = {}
        for r in result:
            anomaly_type = r["dataset"]
            if anomaly_type not in anomaly_dict:
                anomaly_dict[anomaly_type] = []
            anomaly_dict[anomaly_type].append(r["auroc"])

        auroc_data = [{
            "anomaly_type": anomaly_type,
            "auroc": np.mean(aurocs)
        } for anomaly_type, aurocs in anomaly_dict.items()]

        auroc_df = pl.DataFrame(auroc_data)
        avg_auroc = auroc_df["auroc"].mean()
        avg_row = pl.DataFrame({
            "anomaly_type": ["Average"],
            "auroc": [avg_auroc]
        })
        auroc_df = pl.concat([auroc_df, avg_row])

        auroc_csv = csv_save_path / "auroc_metrics.csv"
        auroc_df.write_csv(auroc_csv)

        LOGGER.info(f"✅ AUROC metrics saved to: {auroc_csv}")

        # AUPRC metrics
        auprc_dict = {}
        for r in result:
            anomaly_type = r["dataset"]
            if anomaly_type not in auprc_dict:
                auprc_dict[anomaly_type] = []
            auprc_dict[anomaly_type].append(r["auprc"])

        auprc_data = [{
            "anomaly_type": anomaly_type,
            "auprc": np.mean(auprcs)
        } for anomaly_type, auprcs in auprc_dict.items()]

        auprc_df = pl.DataFrame(auprc_data)
        avg_auprc = auprc_df["auprc"].mean()
        avg_row = pl.DataFrame({
            "anomaly_type": ["Average"],
            "auprc": [avg_auprc]
        })
        auprc_df = pl.concat([auprc_df, avg_row])

        auprc_csv = csv_save_path / "auprc_metrics.csv"
        auprc_df.write_csv(auprc_csv)

        LOGGER.info(f"✅ AUPRC metrics saved to: {auprc_csv}")

    # Time metrics
    time_dict = {}
    for time_item in time_lst:
        for key, value in time_item.items():
            if key not in time_dict:
                time_dict[key] = []
            time_dict[key].append(value)

    time_data = [{
        "metric": key,
        "time_ms": np.mean(values)
    } for key, values in time_dict.items()]
    time_df = pl.DataFrame(time_data)

    time_csv = csv_save_path / "time_metrics.csv"
    time_df.write_csv(time_csv)

    LOGGER.info(f"✅ Time metrics saved to: {time_csv}")


def plot_curves(metrics_save_path: Path,
                dataset_name: str,
                patchcore_name: str,
                result: list[dict[str, float]]) -> None:
    """Plot ROC and PR curves."""
    anomaly_dict = {}
    for r in result:
        anomaly_type = r["dataset"]
        if anomaly_type not in anomaly_dict:
            anomaly_dict[anomaly_type] = {
                "fpr_list": [],
                "roc_tpr_list": [],
                "auroc_list": [],
                "precision_list": [],
                "prc_tpr_list": [],
                "auprc_list": []
            }
        anomaly_dict[anomaly_type]["fpr_list"].append(r["fpr"])
        anomaly_dict[anomaly_type]["roc_tpr_list"].append(r["roc_tpr"])
        anomaly_dict[anomaly_type]["auroc_list"].append(r["auroc"])
        anomaly_dict[anomaly_type]["precision_list"].append(r["precision"])
        anomaly_dict[anomaly_type]["prc_tpr_list"].append(r["prc_tpr"])
        anomaly_dict[anomaly_type]["auprc_list"].append(r["auprc"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # ROC Curve
    for anomaly_type, data in anomaly_dict.items():
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []

        for fpr, tpr in zip(data["fpr_list"], data["roc_tpr_list"]):
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auroc = np.mean(data["auroc_list"])

        ax1.plot(mean_fpr, mean_tpr, linewidth=2,
                 label=f'{anomaly_type} (AUC={mean_auroc:.4f})')

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlim([0., 1.])
    ax1.set_ylim([0., 1.])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title(f'ROC Curves - {dataset_name} ({patchcore_name})', fontsize=14)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(alpha=0.3)

    # PR Curve
    for anomaly_type, data in anomaly_dict.items():
        mean_recall = np.linspace(0, 1, 100)
        precisions = []

        for recall, precision in zip(data["prc_tpr_list"], data["precision_list"]):
            precision_interp = np.interp(mean_recall, recall[::-1], precision[::-1])
            precisions.append(precision_interp)

        mean_precision = np.mean(precisions, axis=0)
        mean_auprc = np.mean(data["auprc_list"])

        ax2.plot(mean_recall, mean_precision, linewidth=2,
                 label=f'{anomaly_type} (AUC={mean_auprc:.4f})')

    ax2.set_xlim([0., 1.])
    ax2.set_ylim([0., 1.])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title(f'PR Curves - {dataset_name} ({patchcore_name})', fontsize=14)
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    curves_file = metrics_save_path / 'curves.png'
    plt.savefig(curves_file, dpi=150, bbox_inches='tight')
    plt.close()

    LOGGER.info(f"✅ ROC & PR curves saved to: {curves_file}")


def plot_heatmap(save_path: Path,
                 score: float,
                 image_path: str,
                 mask: np.ndarray,
                 mask_gt: np.ndarray | None = None,
                 alpha: float = 0.5) -> None:
    """Plot the original image, heatmap, overlay image, and ground truth."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title("Anomaly Heatmap", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Overlay image
    axes[2].imshow(overlay)
    axes[2].text(15, 35, f"Score: {score:.4f}",
                 fontsize=8, color='white', weight='bold', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor='none'))
    axes[2].set_title(f"Overlay (Score: {score:.4f})", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Ground truth
    axes[3].imshow(mask_gt, cmap='gray')
    axes[3].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()

    save_file = save_path / (Path(image_path).stem + "_comparison.png")
    plt.savefig(str(save_file), dpi=150, bbox_inches='tight')
    plt.close()
