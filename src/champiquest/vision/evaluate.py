"""Évaluation du modèle de reconnaissance : métriques + graphes seaborn + sécurité.

Responsabilités :
- prédire logits/labels sur un loader ;
- métriques : accuracy, macro-F1, précision/rappel par classe ;
- matrice de confusion (seaborn heatmap) ;
- **taux d'erreur dangereuse** : fraction des toxiques/mortels prédits comestibles ;
- calibration (temperature scaling) + décision à coût (cf. `decision.py`) ;
- graphes enregistrés dans `reports/` et loggés dans MLflow.

Usage :
    python -m champiquest.vision.evaluate --checkpoint data/models/vision_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from . import datasets, models
from .decision import CostDecision, calibrate_temperature

REPO = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO / "reports"


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (logits, labels) sur tout un loader."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def dangerous_error_rate(y_true: np.ndarray, y_pred: np.ndarray, edible_idx: set, dangerous_idx: set) -> float:
    """Fraction des toxiques/mortels réels prédits comme comestibles (argmax)."""
    dangerous_mask = np.isin(y_true, list(dangerous_idx))
    if dangerous_mask.sum() == 0:
        return 0.0
    edible_pred = np.isin(y_pred, list(edible_idx))
    return float((dangerous_mask & edible_pred).sum()) / dangerous_mask.sum()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize="true")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    ax.set_xlabel("prédit")
    ax.set_ylabel("réel")
    ax.set_title("Matrice de confusion (normalisée par ligne)")
    plt.tight_layout()
    return fig


def plot_per_class_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> plt.Figure:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    df = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=class_names)
    fig, ax = plt.subplots(figsize=(12, 8))
    df[["precision", "recall", "f1"]].plot(kind="bar", ax=ax)
    ax.set_title("Métriques par classe")
    ax.set_ylabel("score")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    return fig


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = models.build_model(ckpt["backbone"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="data/models/vision_best.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint introuvable : {ckpt_path}")

    device = args.device if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(ckpt_path)
    model = model.to(device)

    config = ckpt["config"]
    classes: list[str] = ckpt["classes"]
    danger_of: dict[str, str] = ckpt["danger"]
    num_classes = len(classes) + 1

    _, val_dl, test_dl, _ = datasets.build_dataloaders(
        config=config, seed=config["train"]["seed"]
    )

    # calibration sur le split de validation
    val_logits, val_labels = predict(model, val_dl, device)
    T = calibrate_temperature(val_logits, val_labels)
    print(f"température calibrée : {T:.3f}")

    # test
    test_logits, test_labels = predict(model, test_dl, device)
    probs_cal = softmax(test_logits, T)
    y_pred_raw = probs_cal.argmax(1)

    acc = accuracy_score(test_labels, y_pred_raw)
    f1 = f1_score(test_labels, y_pred_raw, average="macro")

    edible_idx = {i for i, c in enumerate(classes) if danger_of[c] == "comestible"}
    dangerous_idx = {i for i, c in enumerate(classes) if danger_of[c] in ("toxique", "mortel")}

    dec = CostDecision(
        classes,
        danger_of,
        no_mushroom_idx=num_classes - 1,
        cost_edible_as_toxic=config["decision"]["cost_edible_as_toxic"],
        cost_toxic_as_edible=config["decision"]["cost_toxic_as_edible"],
        abstain_threshold=config["decision"]["abstain_threshold"],
    )
    decisions = [dec.decide(p) for p in probs_cal]

    # erreur dangereuse : argmax brut vs décision à coût
    err_raw = dangerous_error_rate(test_labels, y_pred_raw, edible_idx, dangerous_idx)
    edible_claim = np.array([d["decision"] == "edible" for d in decisions])
    dangerous_actual = np.isin(test_labels, list(dangerous_idx))
    err_cost = float((edible_claim & dangerous_actual).sum()) / max(1, dangerous_actual.sum())
    n_abstain = sum(1 for d in decisions if d["decision"] == "abstain")

    class_names = classes + ["no_mushroom"]

    # graphes
    fig_cm = plot_confusion_matrix(test_labels, y_pred_raw, class_names)
    fig_report = plot_per_class_report(test_labels, y_pred_raw, class_names)
    fig_cm.savefig(REPORTS_DIR / "partie2_confusion_matrix.png", dpi=100, bbox_inches="tight")
    fig_report.savefig(REPORTS_DIR / "partie2_per_class_metrics.png", dpi=100, bbox_inches="tight")

    mlflow.set_experiment("partie2-vision")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_params({"temperature": T, "checkpoint": str(ckpt_path)})
        mlflow.log_metrics(
            {
                "test_acc": acc,
                "test_f1": f1,
                "dangerous_error_argmax": err_raw,
                "dangerous_error_cost": err_cost,
                "n_abstain": n_abstain,
            }
        )
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        mlflow.log_figure(fig_report, "per_class_metrics.png")

    print(f"\ntest acc {acc:.3f} | macro-F1 {f1:.3f} | abstentions {n_abstain}/{len(test_labels)}")
    print(f"erreur dangereuse (argmax) : {err_raw:.3f}")
    print(f"erreur dangereuse (coût)   : {err_cost:.3f}")
    print(f"graphes : {REPORTS_DIR / 'partie2_confusion_matrix.png'} + per_class_metrics")


if __name__ == "__main__":
    main()
