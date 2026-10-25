"""Test anti-fuite du split vision : aucun gbifID partagé entre train/val/test.

Pourquoi tester ça : on a justement eu une fuite (22 images dupliquées entre la
classe cible et other_fungus). Verrouiller l'anti-fuite empêche de la réintroduire.
"""
import pytest

from champiquest.vision import datasets

_DATA_PRESENT = (
    datasets.METADATA_CSV.exists()
    and datasets.NEGATIVES_MANIFEST.exists()
    and datasets.OTHER_FUNGUS_MANIFEST.exists()
)

requires_df20 = pytest.mark.skipif(
    not _DATA_PRESENT,
    reason="données DF20/COCO absentes (gitignorées) — test exécuté seulement en local",
)


@requires_df20
def test_aucune_fuite_et_nouvelles_especes_presentes():
    config = datasets.load_config()
    classes, _, _ = datasets.load_classes(config)

    # les 2 espèces ajoutées sont bien dans la config
    assert "Craterellus cornucopioides" in classes
    assert "Hydnum repandum" in classes

    train_df, val_df, test_df = datasets.build_splits(config=config)
    assert datasets.check_no_leakage(train_df, val_df, test_df)
    # aucune image manquante dans le split
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
