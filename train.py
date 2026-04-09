from __future__ import annotations

import argparse
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "dataset"
DATASET_DIR = DATA_ROOT / "dataset"
ZIP_PATH = DATA_ROOT / "Container Damage Detection.yolo26.zip"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_MODEL_NAME = "yolo26s.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_yolo_class():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print("ultralytics is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1) from exc
    return YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26 on the local container-damage dataset.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="Examples: 0, 0,1, cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def default_device() -> str | int:
    try:
        import torch
    except ImportError:
        return "cpu"
    return 0 if torch.cuda.is_available() else "cpu"


def ensure_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_dataset_if_needed() -> None:
    if (DATASET_DIR / "data.yaml").exists():
        return

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Dataset zip not found: {ZIP_PATH}\n"
            "Place the Roboflow zip at dataset/Container Damage Detection.yolo26.zip"
        )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATASET_DIR)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in {path}")
    return data


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def normalize_data_yaml() -> Path:
    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset config: {data_yaml}")

    data = load_yaml(data_yaml)
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"
    write_yaml(data_yaml, data)
    return data_yaml


def collect_pairs(split: str) -> list[tuple[Path, Path]]:
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        return []

    labels = {label.stem: label for label in labels_dir.glob("*.txt")}
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES or not image.is_file():
            continue
        label = labels.get(image.stem)
        if label is not None:
            pairs.append((image, label))
    return pairs


def ensure_split_dirs(split: str) -> tuple[Path, Path]:
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def split_counts(total: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if total < 2:
        raise ValueError("At least 2 labeled images are required to create a train/valid split.")

    val_count = max(1, round(total * val_ratio))
    test_count = round(total * test_ratio) if total >= 5 else 0

    while val_count + test_count >= total:
        if test_count > 0:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            break

    if val_count >= total:
        raise ValueError("Not enough samples left for training after creating validation/test splits.")

    return val_count, max(0, test_count)


def create_missing_splits(val_ratio: float, test_ratio: float, seed: int) -> None:
    train_pairs = collect_pairs("train")
    valid_pairs = collect_pairs("valid")

    if valid_pairs:
        ensure_split_dirs("test")
        return

    if not train_pairs:
        raise FileNotFoundError("No labeled files found under dataset/dataset/train.")

    val_count, test_count = split_counts(len(train_pairs), val_ratio, test_ratio)
    shuffled_pairs = train_pairs[:]
    random.Random(seed).shuffle(shuffled_pairs)

    valid_samples = shuffled_pairs[:val_count]
    test_samples = shuffled_pairs[val_count : val_count + test_count]

    valid_images_dir, valid_labels_dir = ensure_split_dirs("valid")
    test_images_dir, test_labels_dir = ensure_split_dirs("test")

    for image_path, label_path in valid_samples:
        shutil.move(str(image_path), valid_images_dir / image_path.name)
        shutil.move(str(label_path), valid_labels_dir / label_path.name)

    for image_path, label_path in test_samples:
        shutil.move(str(image_path), test_images_dir / image_path.name)
        shutil.move(str(label_path), test_labels_dir / label_path.name)


def prepare_dataset(val_ratio: float, test_ratio: float, seed: int) -> Path:
    extract_dataset_if_needed()
    data_yaml = normalize_data_yaml()
    create_missing_splits(val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    return data_yaml


def load_model(model_name: str) -> "YOLO":
    yolo_class = get_yolo_class()
    local_model_path = MODEL_DIR / model_name
    source = str(local_model_path) if local_model_path.exists() else model_name
    model = yolo_class(source)

    if not local_model_path.exists():
        ckpt_path = getattr(model, "ckpt_path", None)
        if ckpt_path:
            downloaded_path = Path(ckpt_path)
            if downloaded_path.exists():
                shutil.copy2(downloaded_path, local_model_path)
                print(f"Saved pretrained weights to {local_model_path}")

    return model


def copy_best_weight() -> Path:
    best_weight = OUTPUT_DIR / "train" / "weights" / "best.pt"
    if not best_weight.exists():
        raise FileNotFoundError(f"Training completed but best.pt was not found at {best_weight}")

    copied_best = OUTPUT_DIR / "best.pt"
    shutil.copy2(best_weight, copied_best)
    return copied_best


def main() -> None:
    args = parse_args()
    ensure_dirs()

    device = args.device if args.device is not None else default_device()
    data_yaml = prepare_dataset(val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    model = load_model(args.model_name)

    print(f"Dataset: {data_yaml}")
    print(f"Model: {MODEL_DIR / args.model_name if (MODEL_DIR / args.model_name).exists() else args.model_name}")
    print(f"Device: {device}")

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        seed=args.seed,
    )

    best_path = copy_best_weight()
    print(f"Training complete. Best weights copied to {best_path}")


if __name__ == "__main__":
    main()
