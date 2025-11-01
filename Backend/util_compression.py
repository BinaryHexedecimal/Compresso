import torch
from src import MFC
import numpy as np
import networkx as nx
import math
import asyncio
#import pickle


from models import *
from util_data import *
import globals



async def compress_MFC_per_label(label: str,
                            req: CompressRequest,
                            num_per_center=globals.NUM_IMAGE_INSIDE_ETA,
                        ):

    """
    Compress a dataset using Minimum Finite Covering (MFC) per label.
    Supports early cancellation via cancel_callback.
    Works in multiprocessing.Process architecture.
    """
    if req.dataset_name in globals.BUILT_IN_DATASET_NAMES:
        path = f"{globals.DATA_PER_LABEL_DIR}/{req.dataset_name}_percent_{globals.BUILT_IN_DATASET_PERCENT}/{label}.pt"
    else:
        path = f"{globals.DATA_PER_LABEL_DIR}/{req.dataset_name}_percent_{globals.USER_DATASET_PERCENT}/{label}.pt"

    obj = torch.load(path, weights_only=False)

    data_tensor = obj.stacked_tensor
    
    norm_float = globals.NORM_MAP[req.norm]
    mfc_model = MFC(data_tensor, norm=norm_float)

    if req.dataset_name in globals.BUILT_IN_DATASET_NAMES:
        subdir = f"{req.dataset_name}_percent_{globals.BUILT_IN_DATASET_PERCENT}"
    else:
        subdir = f"{req.dataset_name}_percent_{globals.USER_DATASET_PERCENT}"

    # Folder where the .pt file will be stored
    A_dir = os.path.join(globals.ADJ_MATRIX_DIR, subdir)

    # Ensure the subfolder exists
    os.makedirs(A_dir, exist_ok=True)

    # File path for this label and norm
    A_path = os.path.join(A_dir, f"norm_{req.norm}_label_{label}.pt")

    await asyncio.sleep(0.3)
    if globals.ACTIVE_JOBS["compression"][req.compression_job_id]["cancel"]:
        print(f"Compression is cancelled when label {label} is compressed, before calculate A ")
        return None, None, None

    if not os.path.exists(A_path):
        # Compute and save
        A = mfc_model.distanceMatrix()
        torch.save(A, A_path)
        print(f"Matrix A computed and saved to {A_path}")
    else:
        # Load existing tensor
        A = torch.load(A_path)
        print(f"Matrix A loaded from {A_path}")

    await asyncio.sleep(0.3)
    if globals.ACTIVE_JOBS["compression"][req.compression_job_id]["cancel"]:
        print(f"Compression is cancelled when label {label} is compressed, before compreesion ")
        return None, None, None
    compressed_subset, final_eta, sol, t_total = mfc_model.gen_data(A, eta=req.eta, k=req.k, solver=req.optimizer)
    final_eta = float(final_eta)

    # satellite nodes images tensor
    center_idx_lst = (torch.nonzero(torch.tensor(sol)).reshape(-1)).tolist()

    satellite_idx_set = set()
    for c in center_idx_lst:
        selected_indices  = select_indices(A,  c , final_eta, num_per_center)
        satellite_idx_set.update(selected_indices)

    # remove overlap with center list
    satellite_idx_set -= set(center_idx_lst)

    satellite_idx_lst = list(satellite_idx_set)

    nodes_lst = center_idx_lst + satellite_idx_lst
    nodes_tensor = data_tensor[nodes_lst].detach().cpu()
    compressed_subset = compressed_subset.detach().cpu() if isinstance(compressed_subset, torch.Tensor) else compressed_subset
    
    await asyncio.sleep(0.3)
    if globals.ACTIVE_JOBS["compression"][req.compression_job_id]["cancel"]:
        print(f"Compression is cancelled when label {label} is compressed, before create G ")
        return None, None, None

    
    
    # G
    mat = to_numpy_matrix(A)
    m = len(nodes_lst)
    submat = mat[np.ix_(nodes_lst, nodes_lst)]
    G = nx.Graph()
    # Add nodes
    G.add_nodes_from(range(m))
    # Add edges based on threshold
    for i in range(m):
        for j in range(i + 1, m):
            if submat[i, j] < final_eta:
                G.add_edge(i, j, weight=float(submat[i, j]))  # ensure float, not numpy.float32
    
    

    #os.makedirs(globals.TMP_DATA_FOR_GRAPH_DIR, exist_ok=True)
    #base = f"{globals.TMP_DATA_FOR_GRAPH_DIR}/{label}_{req.compression_job_id}"

    #graph_path = f"{base}.gpickle"
    #tensor_path = f"{base}_nodes.pt"

    #with open(graph_path, "wb") as f:
    #    pickle.dump(G, f)

    #torch.save(nodes_tensor, tensor_path)

    return compressed_subset, G, nodes_tensor




def to_numpy_matrix(A):
    """Convert A to a 2D numpy array. Accepts torch.Tensor or numpy array."""
    if isinstance(A, torch.Tensor):
        return A.detach().cpu().numpy()
    return np.asarray(A)



def select_indices(matrix, k, threshold, num_samples):
    # Step 1: ensure matrix is a proper 2D NumPy array
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    elif isinstance(matrix, np.ndarray) and isinstance(matrix.flat[0], torch.Tensor):
        # Convert element-wise if matrix is numpy array of tensors
        matrix = np.array([[float(cell) for cell in row] for row in matrix])
    else:
        matrix = np.asarray(matrix)

    # Step 2: get row k
    row = matrix[k]
    # Convert row to proper NumPy array if somehow still tensor
    if isinstance(row, torch.Tensor):
        row = row.detach().cpu().numpy()
    elif isinstance(row, np.ndarray) and isinstance(row.flat[0], torch.Tensor):
        row = np.array([float(cell) for cell in row])

    #Step 3: filter indices < threshold
    valid_indices = np.where(row < threshold)[0]
    if len(valid_indices) == 0:
        return []
    
    # Step 4: sort by distance
    sorted_indices = valid_indices[np.argsort(row[valid_indices])]

    # Step 5: evenly select up to num_samples
    if len(sorted_indices) <= num_samples + 1:
        selected_indices = sorted_indices[1:]
    else:
        step = int(math.floor(len(sorted_indices) / num_samples ))
        selected_indices = [sorted_indices[1 + int(i * step)] for i in range(num_samples)]

    return [int(idx) for idx in selected_indices]



