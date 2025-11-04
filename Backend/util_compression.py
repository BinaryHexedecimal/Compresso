import torch
import numpy as np
import networkx as nx
import math
import time, asyncio
from datetime import datetime

from src import MFC
from models import *
from util_dataset import *
import globals


# Compress data for a single label using MFC, with periodic cancel checks.
async def compress_MFC_per_label(
    label: str,
    req: CompressRequest,
    num_per_center: int = globals.NUM_IMAGE_INSIDE_ETA,
):
    if req.dataset_name in globals.BUILT_IN_DATASET_NAMES:
        path = f"{globals.DATA_PER_LABEL_DIR}/{req.dataset_name}_percent_{globals.BUILT_IN_DATASET_PERCENT}/{label}.pt"
    else:
        path = f"{globals.DATA_PER_LABEL_DIR}/{req.dataset_name}_percent_{globals.USER_DATASET_PERCENT}/{label}.pt"

    obj = torch.load(path, weights_only=False)
    data_tensor = obj.stacked_tensor

    # Initialize MFC model with data and norm setting
    norm_float = globals.NORM_MAP[req.norm]
    mfc_model = MFC(data_tensor, norm=norm_float)

    # Folder and file for the cached or computed adjacency/distance matrix
    if req.dataset_name in globals.BUILT_IN_DATASET_NAMES:
        subdir = f"{req.dataset_name}_percent_{globals.BUILT_IN_DATASET_PERCENT}"
    else:
        subdir = f"{req.dataset_name}_percent_{globals.USER_DATASET_PERCENT}"
    A_dir = os.path.join(globals.ADJ_MATRIX_DIR, subdir)
    os.makedirs(A_dir, exist_ok=True)
    A_path = os.path.join(A_dir, f"norm_{req.norm}_label_{label}.pt")




    # Allow event loop to process cancellation requests
    await asyncio.sleep(0.2) ## mega vigtig!!
    if globals.ACTIVE_JOBS["compression"][req.compression_job_id]["cancel"]:
        return None, None, None

    # Load or compute the distance matrix
    if not os.path.exists(A_path):
        A = mfc_model.distanceMatrix()
        torch.save(A, A_path)
    else:
        A = torch.load(A_path)




    # Perform MFC compression and get selected points and distances
    await asyncio.sleep(0.3)
    if globals.ACTIVE_JOBS["compression"][req.compression_job_id]["cancel"]:
        return None, None, None
    
    compressed_subset, final_eta, sol, t_total = mfc_model.gen_data(
        A,  k=req.k, solver=req.optimizer
    )
    final_eta = float(final_eta)

    # Center nodes chosen by MFC
    center_idx_lst = (torch.nonzero(torch.tensor(sol)).reshape(-1)).tolist()

    # Satellites are additional nearby points within eta-distance
    satellite_idx_set = set()
    for c in center_idx_lst:
        selected_indices = select_indices(A, c, final_eta, num_per_center)
        satellite_idx_set.update(selected_indices)

    # Remove any overlap with centers
    satellite_idx_set -= set(center_idx_lst)
    satellite_idx_lst = list(satellite_idx_set)

    # Construct tensor from center + satellite nodes
    nodes_lst = center_idx_lst + satellite_idx_lst
    nodes_tensor = data_tensor[nodes_lst].detach().cpu()
    compressed_subset = (
        compressed_subset.detach().cpu()
        if isinstance(compressed_subset, torch.Tensor)
        else compressed_subset
    )

    # Build a graph from node distances
    await asyncio.sleep(0.3)
    if globals.ACTIVE_JOBS["compression"][req.compression_job_id]["cancel"]:
        return None, None, None
    mat = to_numpy_matrix(A)
    m = len(nodes_lst)
    submat = mat[np.ix_(nodes_lst, nodes_lst)]
    G = nx.Graph()
    G.add_nodes_from(range(m))
    for i in range(m):
        for j in range(i + 1, m):
            if submat[i, j] < final_eta:
                G.add_edge(i, j, weight=float(submat[i, j]))
    return compressed_subset, G, nodes_tensor



# Main driver for compressing all labels, yielding progress for event-streaming.
async def run_compression_job(req: CompressRequest, labels: list[str]):
    start_time = time.time()
    compressed_data_by_label = {}
    G_by_label = {}
    nodes_tensor_by_label = {}

    for i, label in enumerate(labels):

        await asyncio.sleep(0.2)
        if globals.ACTIVE_JOBS["compression"][req.compression_job_id]["cancel"]:
            yield {"type": "cancelled", "progress": i}
            return

        compressed_subset, G, nodes_tensor = await compress_MFC_per_label(label, req)

        await asyncio.sleep(0.2)
        if globals.ACTIVE_JOBS["compression"][req.compression_job_id]["cancel"]:
            yield {"type": "cancelled", "progress": i}
            return

        compressed_data_by_label[label] = compressed_subset
        G_by_label[label] = G
        nodes_tensor_by_label[label] = nodes_tensor
        yield {"type": "progress", "progress": i}

    # Summary of whole compression job
    summary = {
        "compression_id": req.compression_job_id,
        "dataset_name": req.dataset_name,
        "timestamp": datetime.now(),
        "norm": req.norm,
        "k": req.k,
        "elapsed_seconds": int(time.time() - start_time),
        "labels": labels,
    }

    # Store full result in memory if not cancelled
    if not globals.ACTIVE_JOBS["compression"][req.compression_job_id]["cancel"]:
        globals.ACTIVE_COMPRESSED_DATA_OBJ = CompressedDatasetObj(
            compression_id=req.compression_job_id,
            compressed_data_by_label=compressed_data_by_label,
            G_by_label=G_by_label,
            nodes_tensor_by_label=nodes_tensor_by_label,
            summary=summary,
            offsets_by_label={key: 0 for key in labels},
        )
    yield {"type": "done"}


# Convert torch.Tensor to numpy for graph/selection logic
def to_numpy_matrix(A):
    if isinstance(A, torch.Tensor):
        return A.cpu().numpy()
    return np.asarray(A)


# Select nearby index values under distance threshold; used to sample satellites.
def select_indices(matrix, k, threshold, num_samples):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    matrix = np.asarray(matrix)

    row = matrix[k]
    valid_indices = np.where(row < threshold)[0]
    if len(valid_indices) == 0:
        return []

    sorted_indices = valid_indices[np.argsort(row[valid_indices])]

    # Pick num_samples evenly spaced, excluding the center itself
    if len(sorted_indices) <= num_samples + 1:
        selected_indices = sorted_indices[1:]
    else:
        step = int(math.floor(len(sorted_indices) / num_samples))
        selected_indices = [sorted_indices[1 + int(i * step)] for i in range(num_samples)]
    return [int(idx) for idx in selected_indices]
