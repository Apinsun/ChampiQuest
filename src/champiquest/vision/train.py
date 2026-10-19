"""Boucle de fine-tuning sur GPU (ConvNeXt-Base + tête N+1).

- Optimizer AdamW + scheduler cosinus ;
- mixed precision (AMP) sur CUDA ;
- early stopping sur la macro-F1 de validation ;
- courbes d'entraînement (seaborn) + checkpoint ;
- suivi MLflow (continuité avec la partie 1).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from . import datasets, models

REPO = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO / "data" / "models"
REPORTS_DIR = REPO / "reports"


def compute_class_weights(train_df, num_classes: int) -> torch.Tensor:
    """Poids de classe par fréquence inverse (moyenne = 1) pour le long tail."""
    counts = train_df["label_idx"].value_counts()
    weights = np.ones(num_classes, dtype=np.float32)
    for i in range(num_classes):
        if i in counts.index:
            weights[i] = len(train_df) / (num_classes * counts[i])
    return torch.tensor(weights)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float, float]:
    """Renvoie (loss, accuracy, macro-F1) sur un loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(labels.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return loss, acc, f1


def train(config, epochs, lr, batch_size, backbone, input_size, device, seed):
    """Fine-tuning complet. Renvoie (modèle, best_val_f1, test_dl, test_df, history, classes, danger)."""
    train_dl, val_dl, test_dl, (train_df, val_df, test_df) = datasets.build_dataloaders(
        batch_size=batch_size, input_size=input_size, seed=seed, config=config
    )
    classes, label_to_idx, label_to_danger = datasets.load_classes(config)
    num_classes = datasets.n_classes_total(config)  # espèces + no_mushroom + other_fungus

    model = models.build_model(backbone, num_classes=num_classes, pretrained=True).to(device)

    class_weights = compute_class_weights(train_df, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=(device == "cuda"))

    history = []
    best_f1 = 0.0
    best_state = None
    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, scaler, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_dl, device)
        scheduler.step()
        mlflow.log_metrics(
            {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1},
            step=epoch,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
        )
        print(
            f"epoch {epoch + 1}/{epochs} | train_loss {train_loss:.3f} | "
            f"val_acc {val_acc:.3f} | val_f1 {val_f1:.3f} | {time.time() - t0:.0f}s"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # checkpoint : état + classes + danger + config (pour recharger sans re-entraîner)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / "vision_best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "backbone": backbone,
            "num_classes": num_classes,
            "classes": classes,
            "danger": label_to_danger,
            "config": config,
        },
        ckpt_path,
    )
    mlflow.log_artifact(ckpt_path)
    return model, best_f1, test_dl, test_df, history, classes, label_to_danger


def plot_training_curves(history: list[dict]) -> plt.Figure:
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=df, x="epoch", y="train_loss", ax=axes[0], label="train")
    sns.lineplot(data=df, x="epoch", y="val_loss", ax=axes[0], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    sns.lineplot(data=df, x="epoch", y="val_acc", ax=axes[1], label="accuracy")
    sns.lineplot(data=df, x="epoch", y="val_f1", ax=axes[1], label="macro-F1")
    axes[1].set_title("Validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = datasets.load_config()
    tc, mc = config["train"], config["model"]
    epochs = args.epochs or tc["epochs"]
    lr = args.lr or tc["lr"]
    batch_size = args.batch_size
    backbone = args.backbone or mc["backbone"]
    input_size = args.input_size or mc["input_size"]
    seed = tc["seed"]
    device = args.device if torch.cuda.is_available() else "cpu"

    mlflow.set_experiment("partie2-vision")
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "backbone": backbone,
                "input_size": input_size,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "seed": seed,
            }
        )
        model, best_f1, test_dl, test_df, history, classes, label_to_danger = train(
            config, epochs, lr, batch_size, backbone, input_size, device, seed
        )

        # courbes d'entraînement → reports + MLflow
        fig_curves = plot_training_curves(history)
        fig_curves.savefig(REPORTS_DIR / "partie2_training_curves.png", dpi=100, bbox_inches="tight")
        mlflow.log_figure(fig_curves, "training_curves.png")
        plt.close(fig_curves)

        test_loss, test_acc, test_f1 = evaluate(model, test_dl, device)
        mlflow.log_metrics({"best_val_f1": best_f1, "test_acc": test_acc, "test_f1": test_f1})
        mlflow.pytorch.log_model(model, "model", serialization_format="pickle")
        print(f"\n✔ terminé (run {run.info.run_id})")
        print(f"  best val F1 : {best_f1:.3f}")
        print(f"  test acc    : {test_acc:.3f}")
        print(f"  test F1     : {test_f1:.3f}")
        print(f"  test        : {len(test_df)} images")
        print(f"  checkpoint  : {MODELS_DIR / 'vision_best.pt'}")


if __name__ == "__main__":
    main()
