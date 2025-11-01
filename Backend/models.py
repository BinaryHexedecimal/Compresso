from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import torch


class OriginDatasetPerLabel(BaseModel):
    dataset_name: str   
    label: str
    stacked_tensor: torch.Tensor
    #adjMatrix_dict: dict
    model_config = {
        "arbitrary_types_allowed": True
    }


class CompressionSummary(BaseModel):
    compression_id: str 
    dataset_name: str
    timestamp: datetime
    norm: str
    k: int
    elapsed_seconds: int
    labels: list 

class CompressedDatasetObj(BaseModel):
    compression_id: str
    compressed_data_by_label: dict
    G_by_label: dict
    nodes_tensor_by_label: dict
    summary: CompressionSummary
    offsets_by_label: dict # for visualisation

    

class CompressRequest(BaseModel):
    compression_job_id:str
    dataset_name: str 
    k: int
    eta: float
    norm: str
    optimizer: str


class StartCompressionResponse(BaseModel):
    success: bool
    message: str


class SavedModelInfo(BaseModel):
    model_id: str
    dataset_name: str
    k: int
    kind: str
    test_acc: float


class BaseTrainRequest(BaseModel):
    train_job_id:str
    kind: str  # "standard" or "adversarial"
    data_info : dict
    user_email: Optional[str] = None 
    optimizer: str = "SGD"
    num_iterations: int = 10
    learning_rate: float = 0.01
    require_adv_attack_test:bool = False


class StandardTrainRequest(BaseTrainRequest):
    pass


class AdvTrainRequest(BaseTrainRequest):
    attack: str = "PGD-linf"
    epsilon: float = 0.3
    alpha: float = 0.01


