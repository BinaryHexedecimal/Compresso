from pydantic import BaseModel
from datetime import datetime
#from typing import Optional
import torch


# Represents a single label's data from a dataset.
# Contains raw tensors for that label after preprocessing.
class OriginDatasetPerLabel(BaseModel):
    dataset_name: str     # Name of the dataset (e.g., 'mnist', 'cifar10')
    label: str            # Class label
    stacked_tensor: torch.Tensor  # PyTorch tensor of samples for this label
    model_config = {
        "arbitrary_types_allowed": True
    } # Allow non-serializable types like torch.Tensor


# Summary metadata for a completed compression job.
# Used for displaying results and saving info in container.
class CompressionSummary(BaseModel):
    compression_id: str      # Unique ID for this compression job
    dataset_name: str        # Name of the dataset compressed
    timestamp: datetime      # When the compression completed
    norm: str                # Norm used (e.g., 'L2', 'Linf')
    k: int                   # Subset size per label (compressed size)
    elapsed_seconds: int     # Time taken to complete compression
    labels: list             # List of dataset labels included 


# In-memory representation of a compressed dataset.
# Contains compressed tensors, graph objects, and summary metadata.
class CompressedDatasetObj(BaseModel):
    compression_id: str
    compressed_data_by_label: dict   # Label-wise tensor data
    G_by_label: dict                 # Label-wise graphs 
    nodes_tensor_by_label: dict      # Label-wise nodes in graph
    summary: CompressionSummary      # Metadata about the compression job
    offsets_by_label: dict           # Used for sequential sampling during visualization


# Request sent from the frontend to trigger data compression.
# Contains settings and parameters for running the compression algorithm.
class CompressRequest(BaseModel):
    compression_job_id: str  # Unique ID for this job 
    dataset_name: str        # Dataset to compress
    k: int                   # Target compressed size per label
    #eta: float               # Radius threshold (graph compression parameter)
    norm: str                # Norm type ('L1', 'L2', 'LINF')
    optimizer: str           # Optimizer (e.g., 'Gurobi', 'CBC')


# Simple response object to indicate if a compression starts.
class StartCompressionResponse(BaseModel):
    success: bool            # Whether the compression job was started
    message: str             # Status or error details


# Metadata stored alongside a trained model, for UI display etc.
class SavedModelInfo(BaseModel):
    model_id: str            # Unique model identifier
    dataset_name: str        # Dataset that the model was trained on
    k: int                   # The size of the compressed dataset that the model was trained on
    kind: str                # Training type ('standard' or 'adversarial')
    test_acc: float          # Accuracy on test data in the final epoch


# Base model for training requests. Used for both standard and adversarial training.
class BaseTrainRequest(BaseModel):
    train_job_id: str             # Unique job ID used for tracking/cancellation
    kind: str                     # "standard" or "adversarial"
    data_info: dict               # The info of compressed dataset used in this training
    #user_email: Optional[str] = None  # Optional email for notifications or logging
    optimizer: str = "SGD"        # Optimizer used in PyTorch (default: SGD)
    num_iterations: int = 10      # Number of training iterations/epochs
    learning_rate: float = 0.01   # Learning rate for training
    require_adv_attack_test: bool = False  # Whether to test with adversarial attacks in the final epoch


# Request for standard training.
class StandardTrainRequest(BaseTrainRequest):
    pass  # Inherits all fields from parent without modification


# Request for adversarial training.
# Adds attack hyperparameters on top of BaseTrainRequest.
class AdvTrainRequest(BaseTrainRequest):
    attack: str = "PGD-linf"       # Attack method 
    epsilon: float = 0.3           # Perturbation bound for adversarial samples
    alpha: float = 0.01            # Step size for PGD attack
