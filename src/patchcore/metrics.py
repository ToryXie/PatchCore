import cv2
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


def evaluate(anomaly_type: str,
             mask: np.ndarray,
             mask_gt: np.ndarray) -> dict[str, str | float]:
    """Evaluator for anomaly detection performance."""
    if mask_gt.max() > 1.:
        mask_gt = cv2.normalize(mask_gt, None, 0, 1, cv2.NORM_MINMAX)

    mask = cv2.resize(mask, (mask_gt.shape[1], mask_gt.shape[0]))
    mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    auroc, fpr, roc_tpr, roc_thresholds = compute_auroc(mask, mask_gt)
    auprc, precision, prc_tpr, prc_thresholds = compute_auprc(mask, mask_gt)

    return {
        "dataset": anomaly_type,
        "auroc": auroc,
        "auprc": auprc,
        "fpr": fpr,
        "precision": precision,
        "roc_tpr": roc_tpr,
        "prc_tpr": prc_tpr,
        "roc_thresholds": roc_thresholds,
        "prc_thresholds": prc_thresholds
    }


def compute_auroc(mask: np.ndarray,
                  masks_gt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the AUROC (Area Under the Receiver Operating Characteristic Curve)."""
    mask_flat = mask.ravel()
    mask_flat_gt = masks_gt.ravel()

    auroc = roc_auc_score(mask_flat_gt, mask_flat)
    fpr, tpr, thresholds = roc_curve(mask_flat_gt, mask_flat)

    return auroc, fpr, tpr, thresholds


def compute_auprc(mask: np.ndarray,
                  masks_gt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the AUPRC (Area Under the Precision-Recall Curve)."""
    mask_flat = mask.ravel()
    mask_flat_gt = masks_gt.ravel()

    precision, recall, thresholds = precision_recall_curve(mask_flat_gt, mask_flat)
    auprc = auc(recall, precision)

    return auprc, precision, recall, thresholds
