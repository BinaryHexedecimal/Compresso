from pathlib import Path
import os, json
from typing import Dict, Optional, Tuple

from util_dataset import *
import globals
import torch


# Ensure a given directory exists, otherwise raise an error.
def ensure_dir_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist")


# Return all JSON summary files in the container directory.
def get_summary_files(container_path: Path):
    return list(container_path.glob("*_summary.json"))


# Check if the container has reached the maximum allowed files.
def is_container_full(container_path: Path, max_files: int) -> Tuple[bool, int]:
    files = get_summary_files(container_path)
    return len(files) >= max_files, len(files)


# Find a duplicate by comparing dataset name, k, and norm in summary files.
def find_duplicate_id(container_path: Path, summary: Dict[str, any]) -> Optional[str]:
    for f in get_summary_files(container_path):
        with open(f, "r") as sf:
            try:
                saved_summary = json.load(sf)
            except Exception:
                continue
            if (
                saved_summary.get("dataset_name") == summary["dataset_name"] and
                saved_summary.get("k") == summary["k"] and
                saved_summary.get("norm") == summary["norm"]
            ):
                return saved_summary.get("compression_id")
    return None


# Remove both JSON summary and .pt data files for a given compression ID.
def remove_compression_by_id(container_path: Path, compression_id: str):
    for suffix in ["_summary.json", "_compressed_data.pt"]:
        path = container_path / f"{compression_id}{suffix}"
        if path.exists():
            os.remove(path)


# Remove the oldest summary (and its data) from the container.
def remove_oldest(container_path: Path) -> Optional[str]:
    summaries = get_summary_files(container_path)
    if not summaries:
        return None
    oldest_file = min(summaries, key=os.path.getmtime)
    oldest_id = oldest_file.stem.replace("_summary", "")
    remove_compression_by_id(container_path, oldest_id)
    return oldest_id


# Save compressed data and its summary into the container.
def save_compression(container_path: Path, obj, compression_id: str):
    all_x, all_y = [], []

    # Map label strings to numeric class indices.
    class_names = load_dataset_classes()[obj.summary.dataset_name]
    label_to_index = {label: idx for idx, label in enumerate(class_names)}

    # Combine all compressed subsets and assign numeric labels.
    for label_str, subset in obj.compressed_data_by_label.items():
        all_x.append(subset)
        all_y.append(torch.full((subset.shape[0],), label_to_index[label_str], dtype=torch.long))

    train_x = torch.cat(all_x, dim=0)
    train_y = torch.cat(all_y, dim=0)

    # Save compressed train data to .pt
    save_path = globals.COMPRESSION_CONTAINER_DIR / f"{compression_id}_compressed_data.pt"
    torch.save({
        "train_x": train_x,
        "train_y": train_y,
        "dataset_name": obj.summary.dataset_name
    }, save_path)

    # Save summary JSON
    with open(container_path / f"{compression_id}_summary.json", "w") as f:
        f.write(obj.summary.model_dump_json())



