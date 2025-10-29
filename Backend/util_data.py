#import random
#from torch.utils.data import TensorDataset, Subset
import json
import os, glob
from PIL import Image
from torchvision import datasets, transforms
import torch


from models import OriginDatasetPerLabel
import globals
#from src import MFC


# ------------------register dataset labels ------------------ #
def load_dataset_classes():
    if os.path.exists(globals.REGISTRY_LABELS_PATH):
        with open(globals.REGISTRY_LABELS_PATH, "r") as f:
            return json.load(f)
    else:
        return {}

def save_dataset_classes(dataset_classes: dict):
    with open(globals.REGISTRY_LABELS_PATH, "w") as f:
        json.dump(dataset_classes, f, indent=4)
    print(f"Saved dataset registry to {globals.REGISTRY_LABELS_PATH}")

def register_dataset_classes(dataset_name: str, class_names: list):
    data = load_dataset_classes()
    if dataset_name in data:
        print(f"Dataset '{dataset_name}' already registered, skipping.")
    else:
        data[dataset_name] = class_names
        save_dataset_classes(data)


# ------------------register active dataset ------------------ #
def load_dataset_names():
    if os.path.exists(globals.REGISTRY_ACTIVE_DATASETS_PATH):
        with open(globals.REGISTRY_ACTIVE_DATASETS_PATH, "r") as f:
            return json.load(f)
    else:
        return {}

def save_dataset_names(active_datasets: list):
    with open(globals.REGISTRY_ACTIVE_DATASETS_PATH, "w") as f:
        json.dump(active_datasets, f, indent=4)
    print(f"Saved dataset registry to {globals.REGISTRY_ACTIVE_DATASETS_PATH}")

def register_dataset_names(dataset_name: str):
    data = load_dataset_names()
    if dataset_name in data: # and data[dataset_name] == class_names:
        print(f"Dataset '{dataset_name}' already registered, skipping.")
    else:
        data.append(dataset_name)
        save_dataset_names(data)


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


# ------------------ pre-process user-defined raw data ------------------

def detect_image_mode_and_size(data_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    if not files:
        raise ValueError(f"No image files found in {data_dir}")
    # assume all images have the same mode and size
    img = Image.open(files[0])
    return img.mode, img.size


def load_user_dataset(dataset_name: str, train_: bool = True):
    """
    Load a user-uploaded dataset from a folder-based structure:
    data_dir/
      ├── train/
      │   ├── class1/
      │   ├── class2/
      └── test/
          ├── class1/
          ├── class2/
    """
    user_data_dir = globals.RAW_DATA_DIR / dataset_name
    if not os.path.exists(user_data_dir):
        raise ValueError("No such a directory for custom user dataset.")
    train_dir = os.path.join(user_data_dir, "train")
    test_dir = os.path.join(user_data_dir, "test")
    base_dir = train_dir if train_ else test_dir
    if not os.path.exists(base_dir):
        raise ValueError(f"Directory not found: {base_dir}")

    mode, size = detect_image_mode_and_size(base_dir)
    if mode == "L": # grayscale
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




# ------------------ load data ------------------


def prepare_train_data(dataset_name:str, percent: int):
    _ = load_dataset(dataset_name, train_ = True)
    create_train_data_obj(dataset_name, percent)


def prepare_test_data(dataset_name:str):
    _ = load_dataset(dataset_name, train_ = False)


def load_dataset(dataset_name: str, train_: bool = True):
    dataset_name = dataset_name.lower()

    TRANSFORM = transforms.ToTensor()

    if dataset_name == "mnist":
        if train_: 
            register_dataset_classes("mnist", [str(i) for i in range(10)])
            register_dataset_names(dataset_name)
        return datasets.MNIST(root=globals.RAW_DATA_DIR  / "MNIST", train=train_, download=True, transform=TRANSFORM)
    elif dataset_name == "cifar10":
        if train_:
            register_dataset_names(dataset_name)
            register_dataset_classes("cifar10", datasets.CIFAR10(root=globals.RAW_DATA_DIR  / "CIFAR10", train=True, download=True).classes)
        return datasets.CIFAR10(root=globals.RAW_DATA_DIR  / "CIFAR10", train=train_, download=True, transform=TRANSFORM)
    elif dataset_name == "cifar100":
        if train_:
            register_dataset_names(dataset_name)
            register_dataset_classes("cifar100", datasets.CIFAR100(root=globals.RAW_DATA_DIR  / "CIFAR100", train=True, download=True).classes)
        return datasets.CIFAR100(root=globals.RAW_DATA_DIR  / "CIFAR100", train=train_, download=True, transform=TRANSFORM)
    elif dataset_name == "svhn":
        if train_:
            register_dataset_names(dataset_name)
            register_dataset_classes("svhn", [str(i) for i in range(10)])
        split = "train" if train_ else "test"
        return datasets.SVHN(root=globals.RAW_DATA_DIR  / "SVHN", split=split, download=True, transform=TRANSFORM)
    else:
        _dataset = load_user_dataset(dataset_name, train_=train_)
        return _dataset



# ------------------------prepare data, sorted by label----------------------#

def create_train_data_obj(dataset_name: str, percent: int) :
    """
    Unified version: handles built-in and user datasets.
    """
    dataset_name = dataset_name.lower()
    out_dir = globals.DATA_PER_LABEL_DIR

    if dataset_name in globals.BUILT_IN_DATASET_NAMES:
        save_path = out_dir/f"{dataset_name}_percent_{globals.BUILT_IN_DATASET_PERCENT}"
    else:
        save_path = out_dir/f"{dataset_name}_percent_{globals.USER_DATASET_PERCENT}"

    if os.path.exists(save_path):
        print(f"Origin data object per label already exists at {save_path}")
    else:

        os.makedirs(save_path, exist_ok=True)
        print(f"Processing dataset: {dataset_name} ({percent}% of data)")

        # --- Load dataset ---
        dataset = load_dataset(dataset_name, train_=True)
        classes = load_dataset_classes()[dataset_name]   # e.g. ['cat','dog','car']

        # --- Initialize class buckets by index ---
        num_classes = len(classes)
        buckets = {i: [] for i in range(num_classes)}

        for img, label in dataset:
            buckets[label].append(img)

        # --- Stack tensors per class, apply percent sampling ---
        for idx, imgs in buckets.items():
            orig_count = len(imgs)
            if orig_count == 0:
                continue

            take_count = max(1, int(orig_count * percent / 100))
            selected_imgs = imgs[:take_count]
            stacked_tensor = torch.stack(selected_imgs)  # shape: (N_class, C, H, W)

            # --- Package into OriginDatasetObj ---
            data_obj = OriginDatasetPerLabel(
                dataset_name=dataset_name,
                stacked_tensor=stacked_tensor,
                label=classes[idx]  # still store the readable names
            )

            # --- Save object ---
            torch.save(data_obj, save_path / f"{classes[idx]}.pt")



def save_trainable_data_in_container(
    compressed_dict: dict,
    compression_job_id: str,
    dataset_name: str):
    all_x = []
    all_y = []

    class_names = load_dataset_classes()[dataset_name]
    label_to_index = {label: idx for idx, label in enumerate(class_names)}

    for label_str, subset in compressed_dict.items():
        all_x.append(subset)
        all_y.append(torch.full((subset.shape[0],), label_to_index[label_str], dtype=torch.long))

    train_x = torch.cat(all_x, dim=0)
    train_y = torch.cat(all_y, dim=0)

    save_path = globals.COMPRESSION_CONTAINER_DIR / f"{compression_job_id}_compressed_data.pt"
    torch.save({
        "train_x": train_x,
        "train_y": train_y,
        "dataset_name": dataset_name
    }, save_path)



def prepare_trainable_data(compressed_dict: dict, dataset_name: str):
    all_x = []
    all_y = []

    class_names = load_dataset_classes()[dataset_name]
    label_to_index = {label: idx for idx, label in enumerate(class_names)}

    for label_str, subset in compressed_dict.items():
        all_x.append(subset)
        all_y.append(torch.full((subset.shape[0],), label_to_index[label_str], dtype=torch.long))

    train_x = torch.cat(all_x, dim=0)
    train_y = torch.cat(all_y, dim=0)

    res = {
        "train_x": train_x,
        "train_y": train_y,
        "dataset_name": dataset_name
    }
    return res
