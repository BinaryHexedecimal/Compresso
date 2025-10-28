from pathlib import Path

BUILT_IN_DATASET_PERCENT = 15
BUILT_IN_DATASET_NAMES = ["mnist", "cifar10", "cifar100", "svhn"]


USER_DATASET_PERCENT = 100



NUM_IMAGE_INSIDE_ETA = 7
MAX_FILES_IN_CONTAINER = 100


BASE_DIR = Path(__file__).resolve().parent  

RAW_DATA_DIR = BASE_DIR / "data" / "raw_data"
COMPRESSION_CONTAINER_DIR = BASE_DIR / "data" / "compression_container"
DATA_PER_LABEL_DIR = BASE_DIR / "data" / "data_per_label"
PERMANENT_TRAIN_MODELS_DIR = BASE_DIR / "data" / "permanent_train_models"
ADJ_MATRIX_DIR = BASE_DIR / "data" / "adj_matrix"

TMP_DATA_FOR_GRAPH_DIR = BASE_DIR / "data" / "tmp" / "gragh"
TMP_DATA_OF_COMPRESSION_DIR = BASE_DIR / "data" / "tmp" / "compression"
TMP_TRAIN_CHECKPOINT_DIR = BASE_DIR / "data" / "tmp" / "train"

#TEST_DATA_DIR = BASE_DIR / "data" / "test"

REGISTRY_LABELS_PATH = BASE_DIR / "data" / "dataset_classes.json"
REGISTRY_ACTIVE_DATASETS_PATH = BASE_DIR / "data" / "active_datasets.json"
TRAINING_HISTORY_PATH = BASE_DIR / "data" / "train_history.json"


NORM_MAP = {
    "L1": 1,
    "L2": 2,
    "MAX": float("inf"),
    "INF": float("inf"),
}
