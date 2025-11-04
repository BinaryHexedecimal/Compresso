from pathlib import Path
import os

# ------------------ Dataset Configuration ------------------ #

# Percentage of data used for compression for each built-in dataset (e.g. 15% of MNIST)
BUILT_IN_DATASET_PERCENT = 15

# Names of built-in datasets this backend natively supports
BUILT_IN_DATASET_NAMES = ["mnist", "cifar10", "cifar100", "svhn"]

# Percentage of data used for compression for user-uploaded datasets (default: 100% per dataset)
USER_DATASET_PERCENT = 100


# ------------------ Compression / Training Hyperparameters ------------------ #

# Maximum allowed epsilon for different norms
EPS_LINF = 0.3        # L-infinity norm threshold
EPS_L2 = 1.5          # L2 norm threshold

# Number of samples to extract inside eta ball (used in graph compression context)
NUM_IMAGE_INSIDE_ETA = 7

# Maximum number of compressed dataset files allowed in container before prompting cleanup
MAX_FILES_IN_CONTAINER = 100

# Maximum available CPU (auto-detected)
NUM_CPU = os.cpu_count()


# ------------------ Base Directory and Paths ------------------ #

# Establish a base path relative to this file's location for consistent directory resolution
BASE_DIR = Path(__file__).resolve().parent

# Folders to store data processed or used throughout backend operations:

# Raw unprocessed dataset files
RAW_DATA_DIR = BASE_DIR / "data" / "raw_data"

# Compressed datasets stored permanently 
COMPRESSION_CONTAINER_DIR = BASE_DIR / "data" / "compression_container"

# Per-label preprocessed dataset tensors
DATA_PER_LABEL_DIR = BASE_DIR / "data" / "data_per_label"

# Saved neural network model files
PERMANENT_TRAIN_MODELS_DIR = BASE_DIR / "data" / "permanent_train_models"

# Adjacency matrices used in compression
ADJ_MATRIX_DIR = BASE_DIR / "data" / "adj_matrix"


# Temporary folders for intermediate computation and state

# Temporarily saved graph data for frontend visualization
#TMP_DATA_FOR_GRAPH_DIR = BASE_DIR / "data" / "tmp" / "gragh"

# Temporary storage for model checkpoints during training (cleared on shutdown or save)
TMP_TRAIN_CHECKPOINT_DIR = BASE_DIR / "data" / "tmp" / "train"


# ------------------ Registry and History Files ------------------ #

# Maps dataset names to label/class names for all known datasets
REGISTRY_LABELS_PATH = BASE_DIR / "data" / "dataset_classes.json"

# List of currently active datasets (built-in or user-uploaded)
REGISTRY_ACTIVE_DATASETS_PATH = BASE_DIR / "data" / "active_datasets.json"

# Track all training runs (job ID, settings, performance metrics, etc.)
TRAINING_HISTORY_PATH = BASE_DIR / "data" / "train_history.json"


# ------------------ Additional Utility Mappings and Globals ------------------ #

# Mapping from string-based norm names to numeric values used in computation
NORM_MAP = {
    "L1": 1,
    "L2": 2,
    "MAX": float("inf"),
    "INF": float("inf"),
}

# Global dictionaries/state for current running jobs and in-memory dataset
ACTIVE_JOBS = {}  # Tracks active jobs (compression or training) with cancel flags
ACTIVE_COMPRESSED_DATA_OBJ = None  # Holds the most recent compressed dataset in memory
