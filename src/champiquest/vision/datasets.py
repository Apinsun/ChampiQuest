"""Chargement des données images : DF20 (espèces) + COCO (négatifs).

Responsabilités :
- lire la config (`configs/vision.yaml`) → 16 classes + danger + paramètres ;
- charger les métadonnées DF20 et les images 300px ;
- charger les négatifs COCO comme classe « pas de champi » ;
- produire un **split par observation** (gbifID) : toutes les images d'une même
  observation vont dans le même split (anti-fuite) ;
- exposer un `Dataset` PyTorch + les `DataLoader` train/val/test.

Exécutable pour vérification :
    python -m champiquest.vision.datasets
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import yaml

REPO = Path(__file__).resolve().parents[3]
DF20_IMG_DIR = REPO / "data" / "raw" / "df20" / "extracted_300px"
METADATA_CSV = REPO / "data" / "raw" / "df20" / "our_classes_metadata.csv"
NEGATIVES_MANIFEST = REPO / "data" / "raw" / "coco" / "negatives_manifest.txt"
CONFIG_PATH = REPO / "configs" / "vision.yaml"

NO_MUSHROOM = "no_mushroom"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Images connues comme mal étiquetées (repérées à l'EDA visuelle) — exclues du dataset.
BLACKLIST = {
    "2237911923-2785.JPG",  # Cantharellus cibarius : photo d'un arbre, pas d'une girolle
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_classes(config: dict | None = None) -> tuple[list[str], dict[str, int], dict[str, str]]:
    """Renvoie (classes, label_to_idx, label_to_danger).

    `classes` = les 16 noms d'espèces (ordre de la config). La classe « pas de
    champi » n'est pas dans `classes` : son index vaut `len(classes)`.
    """
    config = config or load_config()
    classes = [c["name"] for c in config["classes"]]
    label_to_idx = {c: i for i, c in enumerate(classes)}
    label_to_danger = {c["name"]: c["danger"] for c in config["classes"]}
    return classes, label_to_idx, label_to_danger


def load_metadata() -> pd.DataFrame:
    """Charge les métadonnées DF20 de nos classes, hors blacklist."""
    meta = pd.read_csv(METADATA_CSV, dtype=str)
    black = {b.lower() for b in BLACKLIST}
    meta = meta[~meta["image_path"].str.lower().isin(black)]
    return meta


def build_splits(
    config: dict | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split train/val/test par **observation** (gbifID), stratifié par classe.

    - Pour chaque classe : on mélange ses observations (gbifID) et on les répartit
      en test / val / train (proportions fixes, seed). Aucune observation n'est
      répartie entre deux splits → pas de fuite d'observation.
    - Les négatifs COCO (sans observation) sont répartis aléatoirement par image.
    - **Reproductible** : la seed (défaut 42) et les fractions viennent de la config
      (`train.seed`, `train.val_frac`, `train.test_frac`) si non passées en argument.
    """
    config = config or load_config()
    tc = config.get("train", {})
    val_frac = tc.get("val_frac", 0.15) if val_frac is None else val_frac
    test_frac = tc.get("test_frac", 0.15) if test_frac is None else test_frac
    seed = tc.get("seed", 42) if seed is None else seed
    classes, label_to_idx, label_to_danger = load_classes(config)

    meta = load_metadata()
    df = meta.copy()
    df["label_idx"] = df["class_label"].map(label_to_idx)
    # normalisation de casse : metadata .JPG → disque .jpg
    df["file"] = df["image_path"].str.lower().map(lambda n: str(DF20_IMG_DIR / n))

    # split par classe, au niveau observation
    gbif_to_split: dict[str, str] = {}
    for cls in classes:
        cdf = df[df["class_label"] == cls]
        obs = np.array(sorted(cdf["gbifID"].unique()))
        rng = np.random.default_rng(seed)
        rng.shuffle(obs)
        n = len(obs)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        for g in obs[:n_test]:
            gbif_to_split[g] = "test"
        for g in obs[n_test : n_test + n_val]:
            gbif_to_split[g] = "val"
        for g in obs[n_test + n_val :]:
            gbif_to_split[g] = "train"

    df["split"] = df["gbifID"].map(gbif_to_split)
    df["danger"] = df["class_label"].map(label_to_danger)

    # négatifs COCO : répartition aléatoire par image (70/15/15)
    neg_files = [p.strip() for p in open(NEGATIVES_MANIFEST) if p.strip()]
    rng = np.random.default_rng(seed)
    rng.shuffle(neg_files)
    n = len(neg_files)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    neg_rows = []
    for i, f in enumerate(neg_files):
        split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        neg_rows.append(
            {
                "file": f,
                "label_idx": len(classes),  # index de la classe « pas de champi »
                "gbifID": f"neg_{i}",
                "class_label": NO_MUSHROOM,
                "danger": "negatif",
                "split": split,
            }
        )
    neg_df = pd.DataFrame(neg_rows)

    cols = ["file", "label_idx", "gbifID", "class_label", "danger", "split"]
    full = pd.concat([df[cols], neg_df[cols]], ignore_index=True)
    train_df = full[full["split"] == "train"].reset_index(drop=True)
    val_df = full[full["split"] == "val"].reset_index(drop=True)
    test_df = full[full["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def check_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Vérifie qu'aucune observation (gbifID) n'est présente dans deux splits."""
    tr, va, te = set(train_df["gbifID"]), set(val_df["gbifID"]), set(test_df["gbifID"])
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), (
        "FUITE : une observation apparaît dans plusieurs splits !"
    )
    return True


def build_transforms(input_size: int = 384, is_train: bool = True):
    if is_train:
        return T.Compose(
            [
                T.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize(input_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FungiDataset(Dataset):
    """Dataset d'images (espèces + négatifs) à partir d'un DataFrame de split."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["file"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label_idx"])


def build_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    input_size: int = 384,
    seed: int | None = None,
    config: dict | None = None,
):
    """Construit train/val/test DataLoaders + les DataFrames de split associés."""
    train_df, val_df, test_df = build_splits(config=config, seed=seed)
    train_ds = FungiDataset(train_df, build_transforms(input_size, True))
    val_ds = FungiDataset(val_df, build_transforms(input_size, False))
    test_ds = FungiDataset(test_df, build_transforms(input_size, False))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl, (train_df, val_df, test_df)


if __name__ == "__main__":
    import sys

    config = load_config()
    classes, label_to_idx, label_to_danger = load_classes(config)
    print(f"classes : {len(classes)} + 1 ('{NO_MUSHROOM}') = {len(classes) + 1} sorties")

    train_df, val_df, test_df = build_splits(config=config)
    check_no_leakage(train_df, val_df, test_df)
    print("✔ aucune fuite d'observation (gbifID disjoints entre splits)")

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n--- {name} : {len(part)} images ---")
        cnt = part.groupby("class_label").size().sort_values(ascending=False)
        print(cnt.to_string())

    # vérif fichiers existants
    missing = 0
    for df in (train_df, val_df, test_df):
        for f in df["file"]:
            if not Path(f).exists():
                missing += 1
    print(f"\n✔ fichiers manquants : {missing}")

    sys.exit(0)
