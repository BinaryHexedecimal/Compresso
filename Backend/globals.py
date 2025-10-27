from pathlib import Path

# DATA_DIR = Path("./data/default_origin_data")
# USER_RAW_DATA_DIR = Path("./data/user_raw_data")
# COMPRESSION_CONTAINER_DIR = Path("./data/compression_container")
# TRAINING_HISTORY_DIR = Path("./data/train_history.json")
# TMP_DATA_FOR_GRAPH_DIR = Path("./data/tmp_for_compressed_data")
# TMP_TRAIN_CHECKPOINT_DIR = Path("./data/tmp_for_train")  
# PERMANENT_TRAIN_MODELS_DIR = Path("./data/permanent_train_models")
# TEST_DATA_DIR = Path(f"./data/test")
# DATA_PER_LABEL_DIR = Path("./data/data_per_label")
# REGISTRY_LABELS_PATH = Path("./data/dataset_classes.json")
# REGISTRY_ACTIVE_DATASETS_PATH = Path("./data/active_datasets.json")




NUM_IMAGE_INSIDE_ETA = 7
MAX_FILES_IN_CONTAINER = 100


BASE_DIR = Path(__file__).resolve().parent  # points to /app

#TMP_RAW_DATA_DIR = BASE_DIR / "data" / "tmp_raw_data"
#USER_RAW_DATA_DIR = BASE_DIR / "data" / "user_raw_data"

RAW_DATA_DIR = BASE_DIR / "data" / "raw_data"

COMPRESSION_CONTAINER_DIR = BASE_DIR / "data" / "compression_container"
TMP_DATA_FOR_GRAPH_DIR = BASE_DIR / "data" / "tmp_data_for_gragh"
TMP_DATA_OF_COMPRESSION_DIR = BASE_DIR / "data" / "tmp_data_of_compression"

TMP_TRAIN_CHECKPOINT_DIR = BASE_DIR / "data" / "tmp_for_train"
PERMANENT_TRAIN_MODELS_DIR = BASE_DIR / "data" / "permanent_train_models"
TEST_DATA_DIR = BASE_DIR / "data" / "test"
DATA_PER_LABEL_DIR = BASE_DIR / "data" / "data_per_label"

REGISTRY_LABELS_PATH = BASE_DIR / "data" / "dataset_classes.json"
REGISTRY_ACTIVE_DATASETS_PATH = BASE_DIR / "data" / "active_datasets.json"
TRAINING_HISTORY_PATH = BASE_DIR / "data" / "train_history.json"


NORM_MAP = {
    "L1": 1,
    "L2": 2,
    "MAX": float("inf"),
    "INF": float("inf"),
}
