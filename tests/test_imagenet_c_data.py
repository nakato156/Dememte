import json
from pathlib import Path

from PIL import Image

from dememte.config import E5Config, FlowersLegacyConfig, resolve_data_dir
from dememte.data import (
    ImageNetCDataset,
    ImageNetFolderDataset,
    build_imagenet_c_loader,
    build_imagenet_c_loaders,
    build_imagenet_loaders,
)


def _write_image(path: Path, color=(128, 64, 32)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=color).save(path)


def _write_class_index(path: Path):
    payload = {
        "0": ["n00000001", "class_zero"],
        "7": ["n00000008", "class_seven"],
        "999": ["n99999999", "class_last"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_tree(tmp_path: Path):
    root = tmp_path / "imagenet-c-subset"
    for idx in range(5):
        _write_image(root / "gaussian_noise" / "3" / "n00000001" / f"a_{idx}.JPEG")
    for idx in range(3):
        _write_image(root / "gaussian_noise" / "3" / "n00000008" / f"b_{idx}.JPEG")
    _write_image(root / "gaussian_noise" / "5" / "n00000001" / "a.JPEG")
    _write_image(root / "pixelate" / "3" / "n99999999" / "c.JPEG")
    _write_image(root / "pixelate" / "5" / "n99999999" / "c.JPEG")
    mapping = tmp_path / "imagenet_class_index.json"
    _write_class_index(mapping)
    return root, mapping


def test_imagenet_c_dataset_parses_condition_and_labels(tmp_path):
    root, mapping = _make_tree(tmp_path)

    dataset = ImageNetCDataset(
        root,
        "gaussian_noise",
        3,
        class_index_path=mapping,
        download_class_index=False,
    )

    assert len(dataset) == 8
    assert sorted({target for _, target in dataset.samples}) == [0, 7]


def test_imagenet_c_subset_is_deterministic_per_class(tmp_path):
    root, mapping = _make_tree(tmp_path)

    a = ImageNetCDataset(
        root,
        "gaussian_noise",
        3,
        class_index_path=mapping,
        download_class_index=False,
        max_samples_per_class=2,
        seed=123,
    )
    b = ImageNetCDataset(
        root,
        "gaussian_noise",
        3,
        class_index_path=mapping,
        download_class_index=False,
        max_samples_per_class=2,
        seed=123,
    )

    assert len(a) == 4
    assert [str(path) for path, _ in a.samples] == [str(path) for path, _ in b.samples]


def test_imagenet_c_loader_and_suite_metadata(tmp_path):
    root, mapping = _make_tree(tmp_path)

    loader, meta = build_imagenet_c_loader(
        root,
        "pixelate",
        5,
        batch_size=1,
        num_workers=0,
        class_index_path=mapping,
    )
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[-2:] == (224, 224)
    assert batch_y.tolist() == [999]
    assert meta["dataset"] == "imagenet_c"
    assert meta["num_classes"] == 1000

    loaders, suite_meta = build_imagenet_c_loaders(
        root,
        corruptions=["gaussian_noise", "pixelate"],
        severities=[3, 5],
        batch_size=1,
        num_workers=0,
        class_index_path=mapping,
    )
    assert ("gaussian_noise", 3) in loaders
    assert ("pixelate", 5) in loaders
    assert suite_meta["dataset"] == "imagenet_c"


def test_config_defaults_to_imagenet_and_flowers_is_legacy(tmp_path):
    imagenet_root = tmp_path / "imagenet-c-subset"
    (imagenet_root / "gaussian_noise").mkdir(parents=True)
    flowers_root = tmp_path / "flowers-data"
    (flowers_root / "flowers-102").mkdir(parents=True)

    cfg = E5Config(data_dir=str(imagenet_root))
    assert cfg.dataset == "imagenet_c"
    assert cfg.num_classes == 1000
    assert cfg.backbone_name == "resnet50"
    assert cfg.backbone_out_channels == 2048
    assert resolve_data_dir(cfg) == str(imagenet_root.resolve())

    legacy = FlowersLegacyConfig(data_dir=str(flowers_root))
    assert legacy.dataset == "flowers102"
    assert legacy.num_classes == 102
    assert legacy.backbone_name == "resnet18"
    assert resolve_data_dir(legacy) == str(flowers_root.resolve())


def test_clean_imagenet_loader_uses_canonical_class_indices(tmp_path):
    root = tmp_path / "imagenet"
    mapping = tmp_path / "imagenet_class_index.json"
    _write_class_index(mapping)
    _write_image(root / "train" / "n00000008" / "train_a.JPEG")
    _write_image(root / "train" / "n00000001" / "train_b.JPEG")
    _write_image(root / "val" / "n99999999" / "val_a.JPEG")

    train_ds = ImageNetFolderDataset(
        root,
        "train",
        class_index_path=mapping,
        download_class_index=False,
    )
    assert sorted(target for _, target in train_ds.samples) == [0, 7]

    train_loader, val_loader, meta = build_imagenet_loaders(
        root,
        batch_size=1,
        num_workers=0,
        class_index_path=mapping,
    )
    _, y_train = next(iter(train_loader))
    _, y_val = next(iter(val_loader))
    assert int(y_train.item()) in {0, 7}
    assert y_val.tolist() == [999]
    assert meta["dataset"] == "imagenet"
    assert meta["num_classes"] == 1000
    assert meta["train_size"] == 2
    assert meta["val_size"] == 1
