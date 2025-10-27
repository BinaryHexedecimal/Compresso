#from torch.utils.data import Dataset, DataLoader
#import torch.nn as nn
#import torch.nn.functional as F
#from torchvision import datasets, transforms
#import plotly.graph_objects as go
#import sys

import torch
from src import MFC
import multiprocessing
import time
import numpy as np
import networkx as nx
import math
import pickle


from models import *
from util_data import *
import globals



def compress_MFC_per_label(label: str,
                           #data_tensor: torch.Tensor,
                           # A: torch.Tensor,
                            req: CompressRequest,
                            cancel_callback=lambda: False,
                            num_per_center=globals.NUM_IMAGE_INSIDE_ETA,
                        ):

    """
    Compress a dataset using Minimum Finite Covering (MFC) per label.
    Supports early cancellation via cancel_callback.
    Works in multiprocessing.Process architecture.
    """
    print(f"has entered MFC_per_label???")
    path = f"{globals.DATA_PER_LABEL_DIR}/{req.dataset_name}/{label}.pt"
    obj = torch.load(path, weights_only=False)

    data_tensor = obj.stacked_tensor
    A = obj.adjMatrix_dict[req.norm]

    # Early cancel check
    if cancel_callback():
        print("Cancelled before starting MFC")
        return None, None
    norm_float = globals.NORM_MAP[req.norm]
    mfc_model = MFC(data_tensor, norm=norm_float)

    # Cancel before heavy operation
    if cancel_callback():
        print("Cancelled before gen_data")
        return None, None


    print(f"has entered her 1111???")
    compressed_subset, final_eta, sol, t_total = mfc_model.gen_data(A, eta=req.eta, k=req.k, solver=req.optimizer)
    final_eta = float(final_eta)


    print(f"has entered her 2222???")

    # Cancel after heavy operation
    if cancel_callback():
        print("Cancelled after gen_data")
        return None, None
    
    # satellite nodes images tensor
    center_idx_lst = (torch.nonzero(torch.tensor(sol)).reshape(-1)).tolist()

    satellite_idx_set = set()
    for c in center_idx_lst:
        selected_indices  = select_indices(A,  c , final_eta, num_per_center)
        satellite_idx_set.update(selected_indices)
    print(f"has entered her 333333???")
    # remove overlap with center list
    satellite_idx_set -= set(center_idx_lst)

    satellite_idx_lst = list(satellite_idx_set)

    nodes_lst = center_idx_lst + satellite_idx_lst
    print(f"has entered her 444444???")
    print(nodes_lst)
    nodes_tensor = data_tensor[nodes_lst].detach().cpu()
    compressed_subset = compressed_subset.detach().cpu() if isinstance(compressed_subset, torch.Tensor) else compressed_subset
    print(f"has entered her 555555???")
    # G
    mat = to_numpy_matrix(A)
    m = len(nodes_lst)
    submat = mat[np.ix_(nodes_lst, nodes_lst)]
    G = nx.Graph()
    print(f"has entered her 777777???")
    # Add nodes
    G.add_nodes_from(range(m))
    # Add edges based on threshold
    for i in range(m):
        for j in range(i + 1, m):
            if submat[i, j] < final_eta:
                G.add_edge(i, j, weight=float(submat[i, j]))  # ensure float, not numpy.float32
    
    
    
    # save_path = f"{globals.TMP_DATA_FOR_GRAPH_DIR}/visualization_{label}_{req.compression_job_id}.gpickle"


    # with open(save_path, "wb") as f:
    #     pickle.dump(G, f)
    # print(f"has entered her 8888888???")
    # return compressed_subset, nodes_tensor



    os.makedirs(globals.TMP_DATA_FOR_GRAPH_DIR, exist_ok=True)
    base = f"{globals.TMP_DATA_FOR_GRAPH_DIR}/{label}_{req.compression_job_id}"

    graph_path = f"{base}.gpickle"
    tensor_path = f"{base}_nodes.pt"

    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    torch.save(nodes_tensor, tensor_path)

    print(f"✅ Saved {graph_path} and {tensor_path}", flush=True)
    return compressed_subset



def worker_process(req: CompressRequest, 
                    start_time:float,
                    progress_queue: multiprocessing.Queue, 
                    cancel_event: multiprocessing.Event 
                    #result_queue: multiprocessing.Queue
                    ):

    print("begin a worker ?")
    compressed_data_by_label = {}
    #nodes_by_label = {}
    progress_queue.put({"start": True})
    labels = load_dataset_classes()[req.dataset_name]
    total = len(labels)

    for i, label in enumerate(labels):
       
        if cancel_event.is_set():
            progress_queue.put({"cancelled": True, "label": label})
            #result_queue.put(None)   # mark no result
            return
       
        compressed_subset = compress_MFC_per_label(
                                    label, 
                                    req, 
                                    cancel_callback = cancel_event.is_set)
        print(f"for label {label}, finish compress_MFC_label")
        if compressed_subset is None:
            progress_queue.put({"cancelled": True})
            #result_queue.put(None)
            return

        compressed_data_by_label[label] = compressed_subset
        #nodes_by_label[label] = nodes_tensor

        progress_queue.put({
            "progress": i,
            "total": total,
            "label": label
        })

    print(f"begin to make for summary")
    summary = CompressionSummary(
            compression_id = req.compression_job_id,
            dataset_name = req.dataset_name,
            timestamp = datetime.now(),
            norm = req.norm,
            k = req.k,
            elapsed_seconds = int(time.time() - start_time),
            labels = labels,
        )
    
    offsets_by_label = {key: 0 for key in labels}
    
    
    
    
    #result_queue.put((compressed_data_by_label, nodes_by_label, summary, offsets_by_label))
    
    #compressed_data_by_label, nodes_by_label, summary, offsets_by_label = result
    # compressed_dataset_obj =  CompressedDatasetObj(
    #                                 compression_id = req.compression_job_id, 
    #                                 compressed_data_by_label= compressed_data_by_label,
    #                                 #nodes_by_label = nodes_by_label,
    #                                 summary = summary, 
    #                                 offsets_by_label = offsets_by_label,
    #                                 )
    save_path = f"{globals.TMP_DATA_OF_COMPRESSION_DIR}/{req.compression_job_id}_compressed.pt"
    # torch.save({
    #     "compression_id": req.compression_job_id,
    #     "compressed_data_by_label": compressed_data_by_label,
    #     "summary": summary.dict() if hasattr(summary, "dict") else summary.__dict__,
    #     "offsets_by_label": offsets_by_label,
    # }, save_path)

    torch.save({
                "compression_id": req.compression_job_id,
                "compressed_data_by_label": compressed_data_by_label,
                "summary": summary.model_dump(),  # instead of summary.dict()
                "offsets_by_label": offsets_by_label,
                }, save_path)


    progress_queue.put({"done": True})
    
    print(f"summary is ready")
    try:
        progress_queue.close()
    except:
        pass




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

    # Return as Python ints
    return [int(idx) for idx in selected_indices]




# def deep_getsizeof(o, ids=set()):
#     if id(o) in ids:
#         return 0
#     r = sys.getsizeof(o)
#     ids.add(id(o))
#     if isinstance(o, dict):
#         r += sum(deep_getsizeof(k, ids) + deep_getsizeof(v, ids) for k, v in o.items())
#     elif isinstance(o, (list, tuple, set, frozenset)):
#         r += sum(deep_getsizeof(i, ids) for i in o)
#     return r
