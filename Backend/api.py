from fastapi import FastAPI, HTTPException, status, Query, UploadFile, File
from fastapi.responses import StreamingResponse, Response, JSONResponse, FileResponse
import plotly.io as pio
from fastapi.middleware.cors import CORSMiddleware
import json
from contextlib import asynccontextmanager
import os
import torch.multiprocessing as mp
import gurobipy as gp
import torch


import globals
from models import *
from util_train import *
from util_compression import *
from util_dataset import *
from util_container import *
from util_image import *
from src import *

# ------------------ Global setup ------------------

# ACTIVE_JOBS tracks active background tasks like compression and training,
# allowing other endpoints (e.g. cancel actions) to safely access and modify them.
globals.ACTIVE_JOBS = {
    "compression": {},  # e.g. {"job_id": {"cancel": False}}
    "training": {}
}

# ACTIVE_COMPRESSED_DATA_OBJ holds the in-memory result from the recent compression.
globals.ACTIVE_COMPRESSED_DATA_OBJ = None


# ------------------ Unified lifespan ------------------

# FastAPI's lifespan decorator is used to define startup and shutdown behaviors.
# This function initializes global state such as required directories, resets
# temporary data, preloads datasets, and finally performs cleanup during shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup phase ---
    print("Starting backend initialization...")

    # Ensure all required directories exist.
    for d in [
        globals.RAW_DATA_DIR,
        globals.COMPRESSION_CONTAINER_DIR,
        globals.DATA_PER_LABEL_DIR,
        globals.PERMANENT_TRAIN_MODELS_DIR,
        globals.ADJ_MATRIX_DIR,
        globals.TMP_TRAIN_CHECKPOINT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
        if not any(d.iterdir()):
            print(f"Created empty directory: {d}")
        else:
            print(f"Directory exists: {d}")

    # Create base files that act as registries for training progress,
    # active datasets(the datasets that a user has uploaded and not yet deleted), 
    # or label mappings. If the file does not exist,
    # we initialize it with a default JSON structure.
    registry_files = {
        globals.REGISTRY_LABELS_PATH: {},           # Class labels registry
        globals.REGISTRY_ACTIVE_DATASETS_PATH: [],  # Active datasets list
        globals.TRAINING_HISTORY_PATH: [],          # Training runs history
    }

    for path, default_content in registry_files.items():
        if not path.exists():
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(default_content, f, indent=2)
                print(f"Created {path.name} with default content.")
            except Exception as e:
                print(f"Could not create {path.name}: {e}")
        else:
            print(f"{path.name} already exists, skipping.")

    # Clean up temporary directories by deleting all contained files
    for path in [globals.TMP_TRAIN_CHECKPOINT_DIR]:
        if path.exists():
            for file in path.iterdir():
                if file.is_file():
                    file.unlink()
            print(f"Cleared all files in {path}")

    # Preload built-in datasets during startup to make them readily available.
    for id in globals.BUILT_IN_DATASET_NAMES:
        prepare_train_data(id, percent=globals.BUILT_IN_DATASET_PERCENT)
        prepare_test_data(id)

    print("All global state initialized")
    print("Backend startup complete.")

    # Hand control to FastAPI runtime
    yield

    # --- Shutdown phase ---
    # Attempt to terminate all active multiprocessing jobs gracefully 
    # and avoid zombie processes.
    for p in mp.active_children():
        p.terminate()
        print("Terminated one zombie process...")
        p.join()

    print("Cleanup finished. Server shutdown complete.")


# ------------------ FastAPI app setup ------------------

# Instantiate main FastAPI application with the custom lifespan handler.
app = FastAPI(lifespan=lifespan)

# Enable CORS for all origins to allow frontend communication.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------ Compression Features ------------------

# Check if the Gurobi license (a optianol optimization in compression)
# is valid and usable by trying to start its environment.
@app.get("/gurobi-status")
def gurobi_status():
    try:
        print("Checking Gurobi license...")
        env = gp.Env(empty=True)
        env.start()  # this triggers license check
        print("Gurobi environment initialized successfully")
        return {"gurobi_valid": True}
    except gp.GurobiError as e:
        print(f"Gurobi license check failed: {e}")
        return {"gurobi_valid": False}


# Start compression job and stream real-time updates back to client via SSE.
# The compression itself is an async generator (`run_compression_job`) and is
# driven by the `req` parameters. It supports cancellation through `ACTIVE_JOBS`.
@app.post("/compress")
def compress(req: CompressRequest):
    job_id = req.compression_job_id
    labels = load_dataset_classes()[req.dataset_name]

    # Register this job in the global active job tracker.
    globals.ACTIVE_JOBS["compression"][job_id] = {"cancel": False}
    globals.ACTIVE_COMPRESSED_DATA_OBJ = None # realease more memory

    async def event_stream():
        # Announce total number of labels before the loop begins.
        yield f"data: {json.dumps({'type': 'start', 'total': len(labels)})}\n\n"

        # Stream progress updates from the compression job
        async for update in run_compression_job(req, labels):
            yield f"data: {json.dumps(update)}\n\n"

            # Stop streaming after completion or cancellation
            if update.get("type") in ["done", "cancelled"]:
                break

        # Clean up once done or cancelled.
        globals.ACTIVE_JOBS["compression"].pop(job_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Endpoint to cancel a running compression job.
# This sets "cancel" to True in ACTIVE_JOBS, and the running async generator
# periodically checks this flag to exit early.
@app.delete("/cancel_compression/{compression_job_id}")
def cancel_compression(compression_job_id: str):
    job = globals.ACTIVE_JOBS["compression"].get(compression_job_id)
    if not job:
        raise HTTPException(404, f"Job {compression_job_id} not found and failed to cancel")
    job["cancel"] = True
    return {"status": "cancel requested", "id": compression_job_id}



# Fetch summary of the latest in-memory compressed dataset.
# This is useful for displaying details like timestamp, dataset name, k, norm, etc. in Compress Page.
@app.get("/fetch_compression_summary_from_memory/{compression_job_id}")
async def fetch_compression_summary_from_memory(compression_job_id: str):
    if globals.ACTIVE_COMPRESSED_DATA_OBJ.compression_id == compression_job_id:
        summary = globals.ACTIVE_COMPRESSED_DATA_OBJ.summary
        return {"summary": summary, "status": "done"}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid compression_job_id '{compression_job_id}'. Active job is '{globals.ACTIVE_COMPRESSED_DATA_OBJ.compression_id}'."
        )




# ------------------ Training Features ------------------ #

# Start training and stream progress updates using SSE.
# Training logic is encapsulated in the async generator `run_training_job`.
# Supports cancellation via ACTIVE_JOBS["training"] flags that the background loop checks.
@app.post("/train")
async def stream_training(req: BaseTrainRequest):
    # Mark job as active with cancel flag set to False
    globals.ACTIVE_JOBS["training"][req.train_job_id] = {"cancel": False}
    globals.ACTIVE_COMPRESSED_DATA_OBJ = None # realease more memory

    async def event_stream():
        # Create a new request object without shadowing the outer `req`
        train_req = (
            StandardTrainRequest(**req.model_dump())
            if req.kind == "standard"
            else AdvTrainRequest(**req.model_dump())
        )

        # Stream status updates from the training process
        async for update in run_training_job(train_req):
            yield f"data: {json.dumps(update)}\n\n"
            if update.get("type") in ["done", "cancelled"]:
                break

        # Clean up when finished or cancelled
        globals.ACTIVE_JOBS["training"].pop(req.train_job_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")



# Cancel a training job: updates the global cancel flag, allowing the async generator
# in `run_training_job` to exit early when it next checks the flag.
@app.delete("/cancel_train/{train_id}")
def cancel_training(train_id: str):
    job = globals.ACTIVE_JOBS["training"].get(train_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training Job {train_id} not found and fail to cancel.")
    job["cancel"] = True   # Signal the cancellation to worker loop
    return {"status": "cancellation requested", "id": train_id}


# Delete intermediate training checkpoint files for a given `trainId`.
# These files are usually saved during training and stored under TMP_TRAIN_CHECKPOINT_DIR.
@app.delete("/delete_checkpoints/{trainId}")
def delete_checkpoints(trainId: str):
    folder_dir = globals.TMP_TRAIN_CHECKPOINT_DIR
    if not folder_dir.exists():
        return JSONResponse(content={"status": "ok", "message": "No checkpoint folder found."})

    deleted_files = 0
    for file in folder_dir.glob(f"{trainId}_epoch_*.pt"):
        try:
            file.unlink()
            deleted_files += 1
        except Exception as e:
            print(f"Could not delete {file}: {e}")

    return JSONResponse(content={
        "status": "success",
        "message": f"Deleted {deleted_files} checkpoint files for {trainId}."
    })



# ------------------ Visualization Features ------------------ #
# Sample raw images from a uncompressed dataset for a given label.
# The .pt files contain tensors per label and are loaded directly from disk.
@app.get("/sample_origin_images")
def sample_origin_images(dataset_name: str, label: str, n: int):
    if dataset_name in globals.BUILT_IN_DATASET_NAMES:
        path = f"{globals.DATA_PER_LABEL_DIR}/{dataset_name}_percent_{globals.BUILT_IN_DATASET_PERCENT}/{label}.pt"
    else:
        path = f"{globals.DATA_PER_LABEL_DIR}/{dataset_name}_percent_{globals.USER_DATASET_PERCENT}/{label}.pt"

    obj = torch.load(path, weights_only=False)
    dataset = obj.stacked_tensor

    if dataset is None or dataset.numel() == 0:
        raise HTTPException(404, detail=f"Dataset for {label} not found")

    # sample RANDOMLY from uncompressed dataset
    _, images = get_images(dataset, n, 0, random_mode=True)
    return transform_images_for_frontend(images)


# Sample images from the in-memory compressed dataset for a given label.
# Also updates the sequential offset so that successive requests return
# the next batch instead of repeating results.
@app.get("/sample_compressed_images")
def sample_compressed_images(compression_job_id: str, label: str, n: int):
    if (globals.ACTIVE_COMPRESSED_DATA_OBJ.compression_id == compression_job_id):
        dataset_by_label = globals.ACTIVE_COMPRESSED_DATA_OBJ.compressed_data_by_label
        sequential_offsets = globals.ACTIVE_COMPRESSED_DATA_OBJ.offsets_by_label
        labels = globals.ACTIVE_COMPRESSED_DATA_OBJ.summary.labels

        if label not in labels:
            raise HTTPException(404, detail=f"Label {label} not found")

        sequential_offset = sequential_offsets[label]
        new_sequential_offset, images = get_images(
            dataset_by_label[label], n, sequential_offset, random_mode=False
        )
        globals.ACTIVE_COMPRESSED_DATA_OBJ.offsets_by_label[label] = new_sequential_offset
        return transform_images_for_frontend(images)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid compression_job_id '{compression_job_id}'. Active job is '{globals.ACTIVE_COMPRESSED_DATA_OBJ.compression_id}'."
        )


# Return a graph representation in JSON format for the compressed dataset label.
# The graph is rendered using Plotly and returned as a JSON object ready for frontend use.
@app.post("/get_graph_json/{compression_job_id}/{label}/{k}")
def get_graph_json(compression_job_id: str, label: str, k: int):
    if (globals.ACTIVE_COMPRESSED_DATA_OBJ.compression_id == compression_job_id):
        G = globals.ACTIVE_COMPRESSED_DATA_OBJ.G_by_label[label]
        fig = draw_graph(G, c=k)
        fig_json = pio.to_json(fig)  # For frontend rendering
        return JSONResponse(content={"fig_json": fig_json})
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid compression_job_id '{compression_job_id}'. Active job is '{globals.ACTIVE_COMPRESSED_DATA_OBJ.compression_id}'."
        )


# Return a single image (node) from the graph representation of a label.
# The node tensors are stored in memory and returned as PNG byte content.
@app.get("/get_node_image/{compression_job_id}/{label}/{node_index}")
def get_node_image(compression_job_id: str, label: str, node_index: int):
    if (globals.ACTIVE_COMPRESSED_DATA_OBJ.compression_id == compression_job_id):
        nodes = globals.ACTIVE_COMPRESSED_DATA_OBJ.nodes_tensor_by_label[label]
        image_tensor = nodes[node_index]
        img_bytes = tensor_to_image_bytes(image_tensor)
        return Response(content=img_bytes, media_type="image/png")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid compression_job_id '{compression_job_id}'. Active job is '{globals.ACTIVE_COMPRESSED_DATA_OBJ.compression_id}'."
        )




# ------------------ Compressed Data Container Management ------------------ #
# Return all available compression summaries stored in the container directory.
@app.get("/summaries_from_container", response_model=list[CompressionSummary])
def get_all_summaries_from_container():
    folder = globals.COMPRESSION_CONTAINER_DIR
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    summaries = []
    for file in folder.glob("*_summary.json"):
        try:
            with open(file) as f:
                data = json.load(f)
                summaries.append(CompressionSummary(**data))
        except Exception as e:
            print(f"Failed to load {file}: {e}")
    return summaries


# Delete a specific compressed dataset from the container by ID.
# This removes both the Tensor (.pt) file and the summary JSON.
@app.delete("/delete_container_data/{compression_job_id}")
def delete_container_items(compression_job_id: str):
    folder = globals.COMPRESSION_CONTAINER_DIR

    if not folder.exists() or not folder.is_dir():
        return []

    summary_path = os.path.join(folder, f"{compression_job_id}_summary.json")
    pt_path = os.path.join(folder, f"{compression_job_id}_compressed_data.pt")

    deleted_files = []
    for path in [summary_path, pt_path]:
        if os.path.exists(path):
            os.remove(path)
            deleted_files.append(os.path.basename(path))

    if not deleted_files:
        raise HTTPException(status_code=404, detail="No files found for given job_id")

    return {"message": f"Deleted: {', '.join(deleted_files)}"}


# Delete all compressed data files in container directory.
@app.delete("/delete_all_container_data")
def delete_all_container_data():
    folder = globals.COMPRESSION_CONTAINER_DIR

    if not folder.exists() or not folder.is_dir():
        return JSONResponse(content={
            "status": "ok",
            "message": f"No container directory found at {folder}."
        })

    try:
        deleted_files = 0
        for file in folder.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                    deleted_files += 1
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")
        return JSONResponse(content={
            "status": "success",
            "message": f"Has deleted {deleted_files} files"
        })

    except Exception as e:
        print(f"Error clearing container data: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "message": f"Failed to clear container data: {e}"
            },
            status_code=500
        )


# Provide a file download of the compressed .pt data file.
# Optionally specify filename to control how it appears in user's download window.
@app.get("/download_compressed_data/{compressionJobId}")
def download_compressed_data(compressionJobId: str, display_name: str | None = Query(default=None)):
    file_path = globals.COMPRESSION_CONTAINER_DIR / f"{compressionJobId}_compressed_data.pt"
    if not file_path.exists():
        return {"error": f"Data {compressionJobId} not found."}

    filename = display_name if display_name else f"{compressionJobId}.pt"
    return FileResponse(file_path, filename=filename, media_type="application/octet-stream")


# Save in-memory compressed dataset to the container directory.
# Handles duplicate and max-files logic via helper functions.
@app.post("/save/{compression_job_id}")
def save_in_container(compression_job_id: str):
    file_dir = globals.COMPRESSION_CONTAINER_DIR
    ensure_dir_exists(file_dir)

    active_obj = globals.ACTIVE_COMPRESSED_DATA_OBJ
    if compression_job_id != active_obj.compression_id:
        raise HTTPException(400, f"Invalid compression_job_id: {compression_job_id}")

    summary = active_obj.summary.dict()

    full, num_files = is_container_full(file_dir, globals.MAX_FILES_IN_CONTAINER)
    if full:
        return JSONResponse({
            "SaveMessage": f"Container full ({num_files}/{globals.MAX_FILES_IN_CONTAINER} files).",
            "RequireUserDecision": True
        })

    dup_id = find_duplicate_id(file_dir, summary)
    if dup_id:
        return JSONResponse({
            "SaveMessage": "Duplicate found.",
            "RequireUserDecision": True,
            "DuplicateId": dup_id
        })

    save_compression(file_dir, active_obj, compression_job_id)
    return JSONResponse({"SaveMessage": "Saved successfully.", "RequireUserDecision": False})


# Handle user's decision to replace a duplicate or remove oldest file.
@app.post("/handle_replace_choice/{compression_job_id}/{duplicate_id}")
@app.post("/handle_replace_choice/{compression_job_id}")
def handle_replace_choice(compression_job_id: str, duplicate_id: Optional[str] = None):
    file_dir = globals.COMPRESSION_CONTAINER_DIR
    ensure_dir_exists(file_dir)

    active_obj = globals.ACTIVE_COMPRESSED_DATA_OBJ
    if active_obj.compression_id != compression_job_id:
        raise HTTPException(404, f"No active compression with id {compression_job_id}")

    # Remove old or duplicate file
    if duplicate_id:
        remove_compression_by_id(file_dir, duplicate_id)
        action_msg = "Replaced duplicate"
    else:
        removed_id = remove_oldest(file_dir)
        action_msg = f"Replaced oldest ({removed_id})" if removed_id else "No previous files found"

    # Now save new compression
    save_compression(file_dir, active_obj, compression_job_id)

    return JSONResponse({
        "SaveMessage": f"Saved successfully. {action_msg}.",
        "RequireUserDecision": False
    })


# ------------------ Dataset Management ------------------ #

# Get a list of all dataset names currently available (built-in + user-uploaded).
@app.get("/get_all_dataset_names", response_model=list[str])
def get_all_dataset_names():
    names = load_dataset_names()
    return names


# Get all dataset labels (class names) for all datasets.
@app.get("/get_all_dataset_labels")
def get_all_dataset_labels():
    labels = load_dataset_classes()
    return labels


# Delete user-uploaded dataset files from disk.
@app.delete("/delete_dataset/{dataset_name}")
def delete_dataset(dataset_name: str):
    success, failures = delete_dataset_files(dataset_name)
    if success == 0:
        raise HTTPException(404, detail=f"No dataset found for '{dataset_name}'")
    if failures:
        return {"message": f"Deleted partially: {success} items removed, failed at: {failures}"}
    return {"message": f"Deleted '{dataset_name}' successfully."}


# Upload a zipped dataset and preprocess it into per-label tensors.
@app.post("/upload")
async def upload_and_preprocess_user_dataset(file: UploadFile = File(...)):
    try:
        result = await handle_user_dataset_upload(file)
        return result
    except HTTPException as exc:
        raise exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


# Reads the count.json file for the specified dataset, which stores the number of samples per label.
# Returns the smallest sample count across all labels.
# This value is used to validate user input for 'k' in compression 
# to ensure it's not larger than the data allows.
@app.get("/fetch_origin_dataset_min_size_per_label/{dataset_name}")
def fetch_origin_dataset_min_size_per_label(dataset_name: str):
    # Build file path based on dataset type
    if dataset_name in globals.BUILT_IN_DATASET_NAMES:
        filepath = f"{globals.DATA_PER_LABEL_DIR}/{dataset_name}_percent_{globals.BUILT_IN_DATASET_PERCENT}/count.json"
    else:
        filepath = f"{globals.DATA_PER_LABEL_DIR}/{dataset_name}_percent_{globals.USER_DATASET_PERCENT}/count.json"

    print(f"Looking for count file at: {filepath}")

    # Check if file exists
    if not os.path.isfile(filepath):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Count file not found for dataset: {dataset_name}"
        )

    # Load and parse JSON
    try:
        with open(filepath, "r") as json_file:
            counts = json.load(json_file)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Count file for {dataset_name} is corrupted or unreadable"
        )

    # Compute minimum value
    try:
        min_value = min(counts.values())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected content in count file for {dataset_name}"
        )
    return min_value



# ------------------ Training History Management ------------------ #

# Fetch complete history of all training runs.
@app.get("/train_history")
def fetch_train_history():
    file_path = globals.TRAINING_HISTORY_PATH
    if not file_path.exists():
        return {"error": "No training history found"}
    try:
        with open(file_path, "r") as f:
            history = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load training history: {e}"}
    return history


# Remove a specific training run from the history file.
@app.delete("/delete_training_run/{train_id}")
def delete_history_items(train_id: str):
    if os.path.exists(globals.TRAINING_HISTORY_PATH):
        with open(globals.TRAINING_HISTORY_PATH, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    if not isinstance(data, list):
        data = [data]

    has_deleted = False
    for i, run in enumerate(data):
        if run.get("train_job_id") == train_id:
            del data[i]
            has_deleted = True
            break

    if not has_deleted:
        raise HTTPException(status_code=404, detail="No files found for given job_id")
    else:
        with open(globals.TRAINING_HISTORY_PATH, "w") as f:
            json.dump(data, f, indent=4)
        return {"message": f"Deleted trainingrun with {train_id}"}


# Delete all saved training history records.
@app.delete("/delete_all_history")
def delete_all_history():
    history_path = globals.TRAINING_HISTORY_PATH
    if not os.path.exists(history_path):
        return JSONResponse(content={
            "status": "ok",
            "message": f"No training history found at {history_path}."
        })

    try:
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)
        return JSONResponse(content={
            "status": "success",
            "message": "Training history file cleared successfully."
        })
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "message": f"Failed to clear history file: {e}"
            },
            status_code=500
        )


# ------------------ Model Management ------------------ #

# Save a model from a specific training run and epoch into the permanent directory.
# Also writes a JSON file with metadata about the saved model.
@app.post("/save_model/{trainId}/{epoch}")
def save_model(trainId: str, epoch: int, info: SavedModelInfo):
    folder_dir = globals.TMP_TRAIN_CHECKPOINT_DIR
    if not folder_dir.exists():
        raise HTTPException(status_code=500, detail=f"Temporary checkpoint directory {folder_dir} does not exist")

    checkpoint_path = folder_dir / f"{trainId}_epoch_{epoch}.pt"
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint for trainId '{trainId}' epoch '{epoch}' not found")

    save_path = globals.PERMANENT_TRAIN_MODELS_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    model_id = info.model_id
    dest_model_path = save_path / f"{model_id}.pt"
    shutil.copy2(checkpoint_path, dest_model_path)

    # Save metadata as JSON
    dest_info_path = save_path / f"{model_id}_info.json"
    with open(dest_info_path, "w", encoding="utf-8") as f:
        json.dump(info.model_dump(), f, indent=4)

    # Delete all temporary checkpoints for this training ID
    for f in folder_dir.glob(f"{trainId}_epoch_*.pt"):
        try:
            f.unlink()
        except Exception as e:
            print(f"Failed to delete {f}: {e}")

    print(f"Deleted checkpoints for {trainId}")

    return JSONResponse(content={
        "status": "success",
        "message": "Model saved."
    })


# Return metadata for all saved models in permanent storage.
@app.get("/get_models_info", response_model=list[SavedModelInfo])
def get_all_model_info():
    folder = globals.PERMANENT_TRAIN_MODELS_DIR
    if not folder.exists() or not folder.is_dir():
        return []
    model_infos = []
    for filename in os.listdir(folder):
        if filename.endswith("_info.json"):
            filepath = folder / filename
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    model_infos.append(SavedModelInfo(**data))
            except Exception as e:
                print(f"Failed to load model info {filename}: {e}")
                continue
    return model_infos


# Delete a saved model and its metadata JSON from permanent storage.
@app.delete("/delete_model/{modelId}")
def delete_model(modelId: str):
    folder_dir = globals.PERMANENT_TRAIN_MODELS_DIR
    if not folder_dir.exists():
        raise HTTPException(status_code=500, detail=f"Permanent model directory {folder_dir} does not exist")

    model_path = folder_dir / f"{modelId}.pt"
    info_path = folder_dir / f"{modelId}_info.json"

    deleted_files = 0
    if model_path.exists():
        try:
            model_path.unlink()
            deleted_files += 1
        except Exception as e:
            print(f"Could not delete {model_path}: {e}")

    if info_path.exists():
        try:
            info_path.unlink()
            deleted_files += 1
        except Exception as e:
            print(f"Could not delete {info_path}: {e}")

    if deleted_files != 2:
        raise HTTPException(status_code=404, detail=f"No saved model found for ID '{modelId}'")

    return JSONResponse(content={
        "status": "success",
        "message": f"Deleted model '{modelId}' and 2 related files."
    })


# Delete all saved models and metadata in one call.
@app.delete("/delete_all_models")
def delete_all_models():
    folder_dir = globals.PERMANENT_TRAIN_MODELS_DIR
    if not folder_dir.exists():
        return JSONResponse(content={
            "status": "ok",
            "message": "No model folder found."
        })

    for file in folder_dir.glob("*"):
        if file.suffix in [".pt", ".json"]:
            try:
                file.unlink()
            except Exception as e:
                print(f"Could not delete {file}: {e}")

    return JSONResponse(content={
        "status": "success",
        "message": f"Deleted all model files."
    })

# Provide downloadable access to a saved model's .pt file.
# The 'display_name' query parameter lets the client specify the download name.
@app.get("/download_model/{model_id}")
def download_model(model_id: str, display_name: str | None = Query(default=None)):
    file_path = globals.PERMANENT_TRAIN_MODELS_DIR / f"{model_id}.pt"
    if not file_path.exists():
        return {"error": f"Model {model_id} not found."}

    # Default to the model_id as filename if no display name is provided
    filename = display_name if display_name else f"{model_id}.pt"
    return FileResponse(file_path, filename=filename, media_type="application/octet-stream")


# ------------------ Local Run (for Development) ------------------ #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)


