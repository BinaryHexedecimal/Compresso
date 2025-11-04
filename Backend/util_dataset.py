import json
import os, glob
from PIL import Image
from torchvision import datasets, transforms
import torch
from pathlib import Path
import shutil
import zipfile
from typing import List, Dict, Tuple
from fastapi import HTTPException


from models import OriginDatasetPerLabel
import globals


# ------------------ Dataset Label Registry ------------------ #
# Load the mapping of dataset names to class labels.
def load_dataset_classes():
    if os.path.exists(globals.REGISTRY_LABELS_PATH):
        with open(globals.REGISTRY_LABELS_PATH, "r") as f:
            return json.load(f)
    else:
        return {}


# Save updated {dataset:classes} mapping back to disk.
def save_dataset_classes(dataset_classes: dict):
    with open(globals.REGISTRY_LABELS_PATH, "w") as f:
        json.dump(dataset_classes, f, indent=4)
    print(f"Saved dataset registry to {globals.REGISTRY_LABELS_PATH}")


# Add a new dataset and its classed to the registry.
def register_dataset_classes(dataset_name: str, class_names: list):
    data = load_dataset_classes()
    if dataset_name in data:
        print(f"Dataset class '{dataset_name}' already registered, skipping.")
    else:
        data[dataset_name] = class_names
        save_dataset_classes(data)


# ------------------ Dataset Name Registry ------------------ #
# Load all active dataset names.
def load_dataset_names():
    if os.path.exists(globals.REGISTRY_ACTIVE_DATASETS_PATH):
        with open(globals.REGISTRY_ACTIVE_DATASETS_PATH, "r") as f:
            return json.load(f)
    else:
        return {}


# Save updated active dataset names.
def save_dataset_names(active_datasets: list):
    with open(globals.REGISTRY_ACTIVE_DATASETS_PATH, "w") as f:
        json.dump(active_datasets, f, indent=4)
    print(f"Saved dataset registry to {globals.REGISTRY_ACTIVE_DATASETS_PATH}")


# Register a dataset name as active.
def register_dataset_names(dataset_name: str):
    data = load_dataset_names()
    if dataset_name in data:
        print(f"Dataset name '{dataset_name}' already registered, skipping.")
    else:
        data.append(dataset_name)
        save_dataset_names(data)


# Remove dataset from active names registry.
def deactive_dataset(dataset_name: str):
    data = load_dataset_names()
    if dataset_name not in data:
        print(f"Dataset '{dataset_name}' not found in registry.")
        return False
    else:
        data.remove(dataset_name)
        save_dataset_names(data)
        print(f"Deleted dataset '{dataset_name}' from registry.")
        return True


# ------------------ User Data Helpers ------------------ #
# Detect image mode (RGB/Grayscale) and dimension from first image in a folder.
def detect_image_mode_and_size(data_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    if not files:
        raise ValueError(f"No image files found in {data_dir}")
    img = Image.open(files[0])
    return img.mode, img.size


# Load a user-defined dataset with train/test folder structure.
def load_user_dataset(dataset_name: str, train_: bool = True):
    user_data_dir = globals.RAW_DATA_DIR / dataset_name
    if not os.path.exists(user_data_dir):
        raise ValueError("No such directory for custom user dataset.")
    train_dir = os.path.join(user_data_dir, "train")
    test_dir = os.path.join(user_data_dir, "test")
    base_dir = train_dir if train_ else test_dir
    if not os.path.exists(base_dir):
        raise ValueError(f"Directory not found: {base_dir}")

    mode, size = detect_image_mode_and_size(base_dir)
    if mode == "L":  # grayscale
        mean, std = [0.5], [0.5]
        resize_to = (28, 28)
    else:
        mean, std = [0.5]*3, [0.5]*3
        resize_to = (32, 32)

    transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=base_dir, transform=transform)

    if train_:
        register_dataset_classes(dataset_name, dataset.classes)
        register_dataset_names(dataset_name)

    return dataset


# ------------------ Load and Prepare Builtin + User Data ------------------ #
# Prepare built-in or user dataset for compression
def prepare_train_data(dataset_name: str, percent: int):
    _ = load_dataset(dataset_name, train_=True)
    split_original_dataset_by_label(dataset_name, percent)


# Prepare the test split for later model testing.
def prepare_test_data(dataset_name: str):
    _ = load_dataset(dataset_name, train_=False)


# Load dataset by name; built-in datasets also register class names as needed.
def load_dataset(dataset_name: str, train_: bool = True):
    dataset_name = dataset_name.lower()
    TRANSFORM = transforms.ToTensor()

    if dataset_name == "mnist":
        if train_:
            register_dataset_classes("mnist", [str(i) for i in range(10)])
            register_dataset_names(dataset_name)
        return datasets.MNIST(root=globals.RAW_DATA_DIR / "MNIST", train=train_, download=True, transform=TRANSFORM)
    elif dataset_name == "cifar10":
        if train_:
            register_dataset_names(dataset_name)
            register_dataset_classes("cifar10", datasets.CIFAR10(root=globals.RAW_DATA_DIR / "CIFAR10", train=True, download=True).classes)
        return datasets.CIFAR10(root=globals.RAW_DATA_DIR / "CIFAR10", train=train_, download=True, transform=TRANSFORM)
    elif dataset_name == "cifar100":
        if train_:
            register_dataset_names(dataset_name)
            register_dataset_classes("cifar100", datasets.CIFAR100(root=globals.RAW_DATA_DIR / "CIFAR100", train=True, download=True).classes)
        return datasets.CIFAR100(root=globals.RAW_DATA_DIR / "CIFAR100", train=train_, download=True, transform=TRANSFORM)
    elif dataset_name == "svhn":
        if train_:
            register_dataset_names(dataset_name)
            register_dataset_classes("svhn", [str(i) for i in range(10)])
        split = "train" if train_ else "test"
        return datasets.SVHN(root=globals.RAW_DATA_DIR / "SVHN", split=split, download=True, transform=TRANSFORM)
    else:
        return load_user_dataset(dataset_name, train_=train_)


# ------------------------ Per-Label Dataset Serialization ----------------------#
# Split dataset into per-label tensors and save to disk for compression use.
def split_original_dataset_by_label(dataset_name: str, percent: int):
    dataset_name = dataset_name.lower()
    out_dir = globals.DATA_PER_LABEL_DIR

    # Built-in and user datasets get different percentage configs
    if dataset_name in globals.BUILT_IN_DATASET_NAMES:
        save_path = out_dir / f"{dataset_name}_percent_{globals.BUILT_IN_DATASET_PERCENT}"
    else:
        save_path = out_dir / f"{dataset_name}_percent_{globals.USER_DATASET_PERCENT}"

    if os.path.exists(save_path):
        print(f"Origin data object per label already exists at {save_path}")
    else:
        os.makedirs(save_path, exist_ok=True)
        print(f"Processing dataset: {dataset_name} ({percent}% of data)")

        dataset = load_dataset(dataset_name, train_=True)
        classes = load_dataset_classes()[dataset_name]

        # Collect samples into per-class buckets
        num_classes = len(classes)
        buckets = {i: [] for i in range(num_classes)}
        for img, label in dataset:
            buckets[label].append(img)

        # Stack and save tensors per label
        num_per_label = {}
        for idx, imgs in buckets.items():
            orig_count = len(imgs)
            if orig_count == 0:
                continue
            take_count = max(1, int(orig_count * percent / 100))
            selected_imgs = imgs[:take_count]
            stacked_tensor = torch.stack(selected_imgs)

            data_obj = OriginDatasetPerLabel(
                dataset_name=dataset_name,
                stacked_tensor=stacked_tensor,
                label=classes[idx]
            )
            torch.save(data_obj, save_path / f"{classes[idx]}.pt")
            num_per_label[classes[idx]] = take_count

        # Save label counts to JSON
        with open(save_path / f"count.json", "w") as json_file:
            json.dump(num_per_label, json_file, indent=4)


# ------------------------ Dataset Deletion ------------------------#
# Delete all files and folders associated with a dataset.
def delete_dataset_files(dataset_name: str) -> Tuple[int, List[str]]:
    
    deactive_dataset(dataset_name)
    
    paths_to_delete = [
        globals.DATA_PER_LABEL_DIR / f"{dataset_name}_percent_{globals.USER_DATASET_PERCENT}",
        globals.RAW_DATA_DIR / dataset_name,
        globals.ADJ_MATRIX_DIR / f"{dataset_name}_percent_{globals.USER_DATASET_PERCENT}"
    ]

    success_count = 0
    failure_paths = []
    for path in paths_to_delete:
        if path.exists():
            try:
                shutil.rmtree(path)
                success_count += 1
            except Exception:
                failure_paths.append(str(path))

    return success_count == len(paths_to_delete), failure_paths


# ------------------------ ZIP Upload and Process ------------------------#
# Check that uploaded file is a .zip.
def validate_zip_upload(filename: str):
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported.")


# Save uploaded file to a temporary folder and return its path.
def save_temp_upload(file, tmp_dir: Path = Path("/tmp")) -> Path:
    temp_path = tmp_dir / file.filename
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path


# Extract ZIP file to a dataset folder under RAW_DATA_DIR.
def extract_zip_to_dataset_folder(temp_zip_path: Path, dataset_name: str) -> Path:
    dataset_folder = globals.RAW_DATA_DIR / dataset_name
    if dataset_folder.exists():
        shutil.rmtree(dataset_folder)  # Overwrite existing version

    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_folder)

    return dataset_folder


# High-level upload handler: save ZIP, extract, preprocess and return status.
async def handle_user_dataset_upload(file) -> Dict[str, str]:
    validate_zip_upload(file.filename)
    temp_path = save_temp_upload(file)
    print(f"Received ZIP file: {temp_path}")

    dataset_name = Path(file.filename).stem
    try:
        extracted_folder = extract_zip_to_dataset_folder(temp_path, dataset_name)
        print(f"Extracted dataset to: {extracted_folder}")
    finally:
        if temp_path.exists():
            os.remove(temp_path)
            print(f"Removed temp file: {temp_path}")

    # Preprocess extracted images for compression and training
    split_original_dataset_by_label(dataset_name, percent=100)

    return {
        "status": "success",
        "message": f"Dataset '{dataset_name}' uploaded and preprocessed successfully"
    }
