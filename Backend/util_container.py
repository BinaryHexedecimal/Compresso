from pathlib import Path
import os, json
from typing import Dict, Optional, Tuple

from util_dataset import *



def ensure_dir_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist")

def get_summary_files(container_path: Path):
    return list(container_path.glob("*_summary.json"))

def is_container_full(container_path: Path, max_files: int) -> Tuple[bool, int]:
    files = get_summary_files(container_path)
    return len(files) >= max_files, len(files)

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

def remove_compression_by_id(container_path: Path, compression_id: str):
    for suffix in ["_summary.json", "_compressed_data.pt"]:
        path = container_path / f"{compression_id}{suffix}"
        if path.exists():
            os.remove(path)

def remove_oldest(container_path: Path) -> Optional[str]:
    summaries = get_summary_files(container_path)
    if not summaries:
        return None
    oldest_file = min(summaries, key=os.path.getmtime)
    oldest_id = oldest_file.stem.replace("_summary", "")
    remove_compression_by_id(container_path, oldest_id)
    return oldest_id



def save_compression(container_path: Path, obj, compression_id: str):
    #save_trainable_data_in_container(obj.compressed_data_by_label, compression_id, obj.summary.dataset_name)

    # compressed_dict: dict,
    # compression_job_id: str,
    # dataset_name: str):
    all_x = []
    all_y = []

    class_names = load_dataset_classes()[obj.summary.dataset_name]
    label_to_index = {label: idx for idx, label in enumerate(class_names)}

    for label_str, subset in obj.compressed_data_by_label.items():
        all_x.append(subset)
        all_y.append(torch.full((subset.shape[0],), label_to_index[label_str], dtype=torch.long))

    train_x = torch.cat(all_x, dim=0)
    train_y = torch.cat(all_y, dim=0)

    save_path = globals.COMPRESSION_CONTAINER_DIR / f"{compression_id}_compressed_data.pt"
    torch.save({
        "train_x": train_x,
        "train_y": train_y,
        "dataset_name": obj.summary.dataset_name
    }, save_path)




    with open(container_path / f"{compression_id}_summary.json", "w") as f:
        f.write(obj.summary.model_dump_json())





