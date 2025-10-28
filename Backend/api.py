from fastapi import FastAPI, HTTPException, status, UploadFile, File, Query
from fastapi.responses import StreamingResponse, Response, JSONResponse, FileResponse  #HTMLResponse
import plotly.io as pio
from fastapi.middleware.cors import CORSMiddleware
import json
import multiprocessing as mp
from contextlib import asynccontextmanager
import asyncio
import pickle
import time
import shutil
import os
import zipfile
import torch
from datetime import datetime




import globals
from models import *
from util_train import *
from util_compression import *
from util_data import *
from util_image import get_images, draw_graph, transform_images_for_frontend, tensor_to_image_bytes
from util_train import *



# ------------------ Global setup ------------------

# Limit PyTorch CPU threads for safe multi-processing
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["OMP_NUM_THREADS"] = "1"

# Set multiprocessing start method safely
# try:
#     mp.set_start_method("spawn", force=True)  # safer for torch + macOS + Docker
# except RuntimeError:
#     # already set by another module
#     pass




# Track cancel flags and active jobs
ACTIVE_JOBS = {
    "compression": {},  # e.g., {"job_id": {"cancel": False}}
    "training": {}
}

#active_compression_summary = None
active_compressed_data_obj = None



# ------------------ Unified lifespan ------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Unified lifecycle: handles startup + shutdown."""

    # --- Startup phase ---
    print("Starting backend initialization...")

    # ✅ Ensure all required directories exist (add this block)
    for d in [
        globals.RAW_DATA_DIR,
        globals.COMPRESSION_CONTAINER_DIR,
        globals.DATA_PER_LABEL_DIR,
        globals.PERMANENT_TRAIN_MODELS_DIR,
        globals.ADJ_MATRIX_DIR,
        globals.TMP_DATA_FOR_GRAPH_DIR,
        #globals.TMP_DATA_OF_COMPRESSION_DIR,
        globals.TMP_TRAIN_CHECKPOINT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
        # Optional: log which ones were created
        if not any(d.iterdir()):
            print(f"Created empty directory: {d}")
        else:
            print(f"Directory exists: {d}")


    # ✅ Ensure key registry files exist
    registry_files = {
        globals.REGISTRY_LABELS_PATH: {},           # dataset_classes.json → empty dict
        globals.REGISTRY_ACTIVE_DATASETS_PATH: [],  # active_datasets.json → empty list
        globals.TRAINING_HISTORY_PATH: [],          # train_history.json → empty list
    }

    for path, default_content in registry_files.items():
        if not path.exists():
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(default_content, f, indent=2)
                print(f"Created {path.name} with default content.")
            except Exception as e:
                print(f"⚠️ Could not create {path.name}: {e}")
        else:
            print(f"{path.name} already exists, skipping.")



    #active_compressed_data_obj=None


    # app.state.compression_jobs = {}
    # app.state.training_jobs = {}
    # app.state.compressed_data_dict = {}

    # Clean temp dirs
    for path in [globals.TMP_DATA_FOR_GRAPH_DIR, 
                  globals.TMP_TRAIN_CHECKPOINT_DIR]:
        if path.exists():
            for file in path.iterdir():
                if file.is_file():
                    file.unlink()
            print(f"Cleared all files in {path}")

    # Preload standard datasets
    for id in globals.BUILT_IN_DATASET_NAMES:
        #preprocess_data_to_obj(id, train_percent=10, test_percent=20)
        #preprocess_data_to_obj(id, train_percent=10, test_percent=10)
        prepare_train_data(id, percent=globals.BUILT_IN_DATASET_PERCENT)
        prepare_test_data(id)
        

    print("✅ All global state initialized")
    print("✅ Backend startup complete.")

    # Hand control to FastAPI runtime
    yield

    # --- Shutdown phase ---
    print("Cleaning up worker processes...")

    # for job_dict_name in ("compression_jobs", "training_jobs"):
    #     if hasattr(app.state, job_dict_name):
    #         job_dict = getattr(app.state, job_dict_name)
    #         for jobid, job in list(job_dict.items()):
    #             proc = job.get("process")
    #             if proc and proc.is_alive():
    #                 print(f"  Terminating job {jobid} (pid={proc.pid})")
    #                 proc.terminate()
    #                 proc.join(timeout=2)

    #             # Close associated queues and events
    #             for key in ("queue", "results", "cancel"):
    #                 obj = job.get(key)
    #                 if obj:
    #                     try:
    #                         obj.close()
    #                     except Exception:
    #                         pass

    #         job_dict.clear()

    # print("Cleanup finished. Server shutdown complete.")


# ------------------ FastAPI app ------------------

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------ Compression  ------------------

@app.post("/compress")
async def compress(req: CompressRequest):
    start_time = time.time()
    job_id = req.compression_job_id
    ACTIVE_JOBS["compression"][job_id] = {"cancel": False}

    labels = load_dataset_classes()[req.dataset_name]

    async def event_stream():
        # Notify start
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        yield f"data: {json.dumps({'total': len(labels)})}\n\n"
        compressed_data_by_label = {}
        for i, label in enumerate(labels):

            # Let FastAPI handle any cancel requests now
            await asyncio.sleep(0.2)


            if ACTIVE_JOBS["compression"][job_id]["cancel"]:
                #yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                break

            compressed_subset = compress_MFC_per_label(label, req)
            compressed_data_by_label[label] = compressed_subset

            yield f"data: {json.dumps({'progress': i})}\n\n"

        if ACTIVE_JOBS["compression"][job_id]["cancel"]:
            yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
        else:
            summary = CompressionSummary(
                compression_id=req.compression_job_id,
                dataset_name=req.dataset_name,
                timestamp=datetime.datetime.now(),
                norm=req.norm,
                k=req.k,
                elapsed_seconds=int(time.time() - start_time),
                labels=labels,
            )

            offsets_by_label = {key: 0 for key in labels}
            global active_compressed_data_obj
            active_compressed_data_obj = CompressedDatasetObj(
                compression_id=req.compression_job_id,
                compressed_data_by_label=compressed_data_by_label,
                summary=summary,
                offsets_by_label=offsets_by_label,
            )

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        ACTIVE_JOBS["compression"].pop(job_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")



@app.delete("/cancel_compression/{compression_job_id}")
def cancel_compression(compression_job_id: str):
    print("will set job cancelled")
    job = ACTIVE_JOBS["compression"].get(compression_job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job["cancel"] = True
    return {"status": "cancel requested", "id": compression_job_id}





# ------------------ Image supports  ------------------
@app.get("/sample_origin_images")
def sample_origin_images(dataset_name: str, label: str, n: int):
    if dataset_name in globals.BUILT_IN_DATASET_NAMES:
        path = f"{globals.DATA_PER_LABEL_DIR}/{dataset_name}_percent_{globals.BUILT_IN_DATASET_PERCENT}/{label}.pt"
    else:
        path = f"{globals.DATA_PER_LABEL_DIR}/{dataset_name}_percent_{globals.USER_DATASET_PERCENT}/{label}.pt"

        
    obj = torch.load(path, weights_only=False)
    dataset_for_one_label = obj.stacked_tensor
    
    if dataset_for_one_label is None or dataset_for_one_label.numel() == 0:
        raise HTTPException(404, detail="Dataset not found")

    _, images = get_images(dataset_for_one_label, n, 0, random_mode = True)
    return transform_images_for_frontend(images)


@app.get("/sample_compressed_images")
def sample_compressed_images(compression_job_id:str, label: str, n: int):
    if (active_compressed_data_obj.compression_id == compression_job_id):
        #compressed_data_obj = app.state.compressed_data_dict[compression_job_id]
        dataset_by_label = active_compressed_data_obj.compressed_data_by_label
        sequential_offsets = active_compressed_data_obj.offsets_by_label
        labels = active_compressed_data_obj.summary.labels
        if label not in labels:
            raise HTTPException(404, detail=f"Label {label} not found")
        sequential_offset = sequential_offsets[label]
        new_sequential_offset, images = get_images(
                dataset_by_label[label], n, sequential_offset, random_mode = False)
        active_compressed_data_obj.offsets_by_label[label] = new_sequential_offset
        return transform_images_for_frontend(images)
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid compression_job_id '{compression_job_id}'. Active job is '{active_compressed_data_obj.compression_id}'."
        )


@app.post("/get_graph_json_data/{compression_job_id}/{label}/{k}")
def get_graph_json_data(compression_job_id: str, label: str, k: int):
    read_path = f"{globals.TMP_DATA_FOR_GRAPH_DIR}/{label}_{compression_job_id}.gpickle"
    with open(read_path, "rb") as f:
        G = pickle.load(f)
    fig = draw_graph(G, c=k)
    fig_json = pio.to_json(fig)  # For frontend rendering
    return JSONResponse(content={"fig_json": fig_json})


@app.get("/get_node_image/{compression_job_id}/{label}/{node_index}")
def get_node_image(compression_job_id: str, label: str, node_index: int):
    read_path = f"{globals.TMP_DATA_FOR_GRAPH_DIR}/{label}_{compression_job_id}_nodes.pt"
    nodes =torch.load(read_path, map_location="cpu", weights_only=False)
    image_tensor = nodes[node_index]
    img_bytes = tensor_to_image_bytes(image_tensor)
    return Response(content=img_bytes, media_type="image/png")

@app.delete("/delete_graph_data/{compressionId}")
def delete_graphs(compressionId: str):
    folder_dir = globals.TMP_DATA_FOR_GRAPH_DIR
    if not folder_dir.exists():
        return JSONResponse(content={"status": "ok", "message": "No graph folder found."})
    deleted_files = 0
    # Match *any* file that contains the compressionId, regardless of extension or prefix
    for file in folder_dir.glob(f"*{compressionId}*"):
        try:
            file.unlink()
            deleted_files += 1
        except Exception as e:
            print(f"Could not delete {file}: {e}")
    return JSONResponse(content={
        "status": "success",
        "message": f"🧹Deleted {deleted_files} graph files for compressionId '{compressionId}'."
    })

@app.delete("/delete_all_graph_data")
def delete_all_graph():
    folder_dir = globals.TMP_DATA_FOR_GRAPH_DIR

    if not folder_dir.exists():
        return JSONResponse(content={
            "status": "ok",
            "message": "No graph folder found."
        })

    deleted_files = 0
    for file in folder_dir.iterdir():
        if file.is_file():
            try:
                file.unlink()
                deleted_files += 1
            except Exception as e:
                print(f"Could not delete {file}: {e}")

    return JSONResponse(content={
        "status": "success",
        "message": f"🧹Deleted {deleted_files} graph files."
    })


# ------------------ Summary supports  ------------------

@app.get("/fetch_compression_summary_from_memory/{compression_job_id}")
async def fetch_compression_summary_from_memory(compression_job_id: str):
    if active_compressed_data_obj.compression_id == compression_job_id:
        summary = active_compressed_data_obj.summary
        return {"summary": summary, "status": "done"}
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid compression_job_id '{compression_job_id}'. Active job is '{active_compressed_data_obj.compression_id}'."
        )



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


@app.get("/summary_from_container/{compression_job_id}", response_model=CompressionSummary)
def get_one_summary_from_container(compression_job_id: str):
    folder = globals.COMPRESSION_CONTAINER_DIR
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail="User container folder not found")

    expected_filename = f"{compression_job_id}_summary.json"
    summary_path = folder / expected_filename

    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"Summary for dataId '{compression_job_id}' not found")

    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
            return CompressionSummary(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load summary: {e}")




# ------------------ Maintain compressed data container and graphs ------------------

@app.delete("/delete_container_data/{compression_job_id}")
def delete_container_items(compression_job_id:str):

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

        # Only delete files (no recursive delete)
        for file in folder.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                    deleted_files += 1
                except Exception as e:
                    print(f"⚠️Failed to delete {file}: {e}")

        return JSONResponse(content={
            "status": "success",
            "message": f"Has deleted {deleted_files} files"
        })

    except Exception as e:
        print(f"❌Error clearing container data: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "message": f"Failed to clear container data: {e}"
            },
            status_code=500
        )




@app.post("/save/{compression_job_id}")
def save_in_container(compression_job_id: str):
    file_dir = globals.COMPRESSION_CONTAINER_DIR
    if not file_dir.exists():
        raise HTTPException(status_code=500, detail=f"Container directory {file_dir} does not exist")
    
    if compression_job_id != active_compressed_data_obj.compression_id:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid compression_job_id '{compression_job_id}'. Active job is '{active_compressed_data_obj.compression_id}'."
        )
    else:
        #compressed_obj = app.state.compressed_data_dict.get(compression_job_id)
        current_summary = active_compressed_data_obj.summary

        dataset_name = current_summary.dataset_name
        
        files = [f for f in os.listdir(file_dir) if f.endswith("_summary.json")]
        if len(files) >= globals.MAX_FILES_IN_CONTAINER :
            return JSONResponse(content={
                "SaveMessage": f"⚠️Container full: {len(files)}/{globals.MAX_FILES_IN_CONTAINER} files.",
                "RequireUserDecision": True
            })
        for f in files:
            with open(os.path.join(file_dir, f), "r") as sf:
                try:
                    saved_summary = json.load(sf)
                except Exception:
                    continue  # skip corrupted or unreadable summaries

                if (saved_summary.get("dataset_name") == current_summary.dataset_name and
                    saved_summary.get("k") == current_summary.k and
                    saved_summary.get("norm") == current_summary.norm):

                    print("Found duplicate id is " + saved_summary.get("compression_id"))
                    return JSONResponse(content={
                        "SaveMessage": (
                            f"⚠️Duplicate found: Dataset {current_summary.dataset_name}, "
                            f"k={current_summary.k}, norm={current_summary.norm} already exists."
                        ),
                        "RequireUserDecision": True,
                        "DuplicateId": saved_summary.get("compression_id")
                    })

        # Save .pt 
        save_trainable_data_in_container(
                                active_compressed_data_obj.compressed_data_by_label, 
                                compression_job_id,
                                dataset_name = dataset_name)

        # Save summary
        #torch.save(data_obj, os.path.join(file_dir, f"{compression_job_id}_compressed_data.pt"))
        with open(os.path.join(file_dir, f"{compression_job_id}_summary.json"), "w") as f:
            f.write(active_compressed_data_obj.summary.json())

        return JSONResponse(content={
            "SaveMessage": "✅Saved successfully.",
            "RequireUserDecision": False})


@app.post("/handle_replace_choice/{compression_job_id}/{duplicate_id}")
@app.post("/handle_replace_choice/{compression_job_id}")  
def handle_replace_choice(compression_job_id: str, duplicate_id: Optional[str] = None):
    file_dir = globals.COMPRESSION_CONTAINER_DIR

    # --- Basic checks ---
    if not file_dir.exists():
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "SaveMessage": f"❌Container directory {file_dir} does not exist.",
                "RequireUserDecision": True
            }
        )
    # print(active_compressed_data_obj.compression_id)
    # print(compression_job_id)
    # print(duplicate_id)

    #compressed_obj = app.state.compressed_data_dict.get(compression_job_id)
    if active_compressed_data_obj.compression_id != compression_job_id:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "SaveMessage": f"❌Compressed data not found for id {compression_job_id}.",
                "RequireUserDecision": True
            }
        )

    try:
        message_suffix = ""  # Will hold message text about what was replaced

        # --- CASE 1: No duplicate_id → find and delete oldest summary ---
        if not duplicate_id:
            print("ℹ️No duplicate_id provided — searching for oldest summary...")
            summaries = list(file_dir.glob("*_summary.json"))

            if summaries:
                oldest_file = min(summaries, key=os.path.getmtime)
                delete_id = oldest_file.stem.replace("_summary", "")
                print(f"🕰️Oldest summary found and will be replaced: {oldest_file.name}")

                # Delete both summary and data files
                for suffix in ["_summary.json", "_compressed_data.pt"]:
                    path = file_dir / f"{delete_id}{suffix}"
                    if path.exists():
                        os.remove(path)
                        print(f"🗑️Deleted old file: {path.name}")

                message_suffix = " by replacing the oldest compression."
            else:
                print("⚠️No summaries found to delete.")
                message_suffix = "\nℹ️No prior compressions found — nothing replaced."

        # --- CASE 2: duplicate_id provided explicitly ---
        else:
            print(f"♻️Replacing specific duplicate: {duplicate_id}")
            for suffix in ["_summary.json", "_compressed_data.pt"]:
                path = file_dir / f"{duplicate_id}{suffix}"
                if path.exists():
                    os.remove(path)
                    print(f"🗑️Deleted duplicate file: {path.name}")

            message_suffix = " by replacing the old duplicate."

        # --- Save new compressed results ---
        dataset_name = active_compressed_data_obj.summary.dataset_name
        save_trainable_data_in_container(active_compressed_data_obj.compressed_data_by_label, 
                                             compression_job_id,
                                             dataset_name=dataset_name)

        
        with open(file_dir / f"{compression_job_id}_summary.json", "w") as f:
            f.write(active_compressed_data_obj.summary.json())

        final_message = f"✅Saved successfully{message_suffix}"
        print(final_message)

        return JSONResponse(
            content={
                "SaveMessage": final_message,
                "RequireUserDecision": False
            }
        )

    except Exception as e:
        print("❌ Replacement failed:", e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "SaveMessage": f"❌ Replacement failed: {str(e)}",
                "RequireUserDecision": False
            }
        )





@app.get("/download_compressed_data/{compressionJobId}")
def download_compressed_data(compressionJobId: str, display_name: str | None = Query(default=None)):
    
    file_path = globals.COMPRESSION_CONTAINER_DIR / f"{compressionJobId}_compressed_data.pt"
    if not file_path.exists():
        return {"error": f"Data {compressionJobId} not found."}

    # sanitize fallback
    filename = display_name if display_name else f"{compressionJobId}.pt"
    return FileResponse(file_path, filename=filename, media_type="application/octet-stream")




# ------------------User Dataset supports  ------------------

@app.get("/get_all_dataset_names", response_model=list[str])
def get_all_dataset_names():
    names = load_dataset_names()
    return names

@app.get("/get_all_dataset_labels")
def get_all_dataset_labels():
    labels = load_dataset_classes()
    return labels


@app.delete("/delete_dataset/{dataset_name}")
def delete_dataset(dataset_name: str):
    """
    Delete dataset entry from registry and remove any related directories.
    """

    # --- Delete from active dataset registry ---
    # --- But keep it in the class registry, because the training history, compressed data may still need it
    success_deregistry = deactive_dataset(dataset_name)

    if not success_deregistry:
        raise HTTPException(status_code=404, detail="No dataset found for given dataset name")

    # --- Define directories to delete ---
    paths_to_delete = [os.path.join(globals.DATA_PER_LABEL_DIR, dataset_name + f"_percent_{globals.USER_DATASET_PERCENT}"),
                        os.path.join(globals.RAW_DATA_DIR, dataset_name),
                        os.path.join(globals.ADJ_MATRIX_DIR, dataset_name) + f"_percent_{globals.USER_DATASET_PERCENT}"
                        ]
    # --- Attempt deletion for each path ---
    success_delete_data = 0
    for path_to_delete in paths_to_delete:
        if os.path.exists(path_to_delete):
            try:
                shutil.rmtree(path_to_delete)
                success_delete_data += 1
                print(f"Deleted directory: {path_to_delete}")
            except Exception as e:
                print(f"Failed to delete {path_to_delete}: {e}")

   
    if success_delete_data == len(paths_to_delete):
        return {"message": f"✅Deleted '{dataset_name}'" }
    else:
        return {"message": f"{dataset_name} cannot be deleted completely, check it" }


@app.post("/upload")
async def upload_and_preprocess_user_dataset(file: UploadFile = File(...)):
    """
    Receive a ZIP dataset file from the frontend, extract it, and store it.
    """
    try:
        # Ensure it's a ZIP file
        if not file.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="Only .zip files are supported.")

        # Save uploaded file temporarily
        temp_path = os.path.join("/tmp", file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Received ZIP file: {temp_path}")

        # Extract dataset name (remove .zip extension)
        dataset_name = os.path.splitext(file.filename)[0]
        dataset_folder = os.path.join(globals.RAW_DATA_DIR, dataset_name)

        # Extract ZIP into the target directory
        with zipfile.ZipFile(temp_path, "r") as zip_ref:
            zip_ref.extractall(dataset_folder)

        # Cleanup the temp file
        os.remove(temp_path)

        print(f"Extracted dataset: {dataset_folder}")


    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.")
    except Exception as e:
        print(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


    try:
        create_train_data_obj(dataset_name, percent=100)
        return {
            "status": "success",
            "message": f"✅ Dataset '{dataset_name}' preprocessed successfully"
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Preprocessing failed: {str(e)}")









# ------------------ Train history supports  ------------------
@app.get("/train_history")
def fetch_train_history():
    file_path = globals.TRAINING_HISTORY_PATH

    if not file_path.exists():
        return {"error": "No training history found"}

    try:
        with open(file_path, "r") as f:
            history = json.load(f)  # this should already be a list
    except Exception as e:
        return {"error": f"Failed to load training history: {e}"}

    return history



@app.delete("/delete_training_run/{train_id}")
def delete_history_items(train_id:str):
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


@app.delete("/delete_all_history")
def delete_all_history():
    """
    Clears (empties) the training history JSON file.
    """
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
            "message": "🧹 Training history file cleared successfully."
        })

    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "message": f"❌ Failed to clear history file: {e}"
            },
            status_code=500
        )


# ------------------ Train  ------------------
@app.post("/train")
async def stream_training(req: BaseTrainRequest):

    #cancel_event = multiprocessing.Event()
    #progress_queue = multiprocessing.Queue()
    ACTIVE_JOBS["training"][req.train_job_id] = {"cancel": False}

    async def event_stream():
        # Notify start

        req_ = req.model_dump()
        kind = req_.get("kind", "").lower()
        if kind == "standard":
            req_obj = StandardTrainRequest(**req_)
        else:
            req_obj = AdvTrainRequest(**req_)

        data_path = globals.COMPRESSION_CONTAINER_DIR / f"{req_obj.data_info.get('data_id')}_compressed_data.pt"
        res = torch.load(data_path, weights_only=False)
        train_dataset = TensorDataset(res["train_x"], res["train_y"])
        #train_loader = DataLoader(data_obj, batch_size=64, shuffle=True)
        test_dataset = load_dataset(req_obj.data_info.get('dataset_name'), train_ = False)

        #progress_queue.put({"type": "start"})
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

        sample, _ = train_dataset[0]
        _in_channels = sample.shape[0] 
        _num_classes = len(load_dataset_classes()[req_obj.data_info["dataset_name"]])
        #print(f"inside training, channael is {_in_channels} , num of class is {_num_classes}")
        model = ConvNet(in_channels=_in_channels, num_classes=_num_classes)

        #eps_linf=0.3,
        #eps_l2=1.5,

        trainer = TR(model, train_loader, test_loader, eps_linf=globals.EPS_LINF, eps_l2=globals.EPS_L2)
        
        # Select optimizer
        if req_obj.optimizer.lower() == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=req_obj.learning_rate, momentum=0.9, weight_decay=5e-4)
        elif req_obj.optimizer.lower() == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=req_obj.learning_rate, weight_decay=5e-4)
        else:
            raise ValueError(f"Unknown optimizer: {req_obj.optimizer}")


        all_epochs = []
        run_info = {"status": "started", 
                             "train_job_id": req_obj.train_job_id,
                             "timestamp": "",
                             "epochs": [], 
                             "req_obj": vars(req_obj)}

        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        for epoch in range(1, req_obj.num_iterations+1):
            await asyncio.sleep(0.2)
            if ACTIVE_JOBS["training"][req_obj.train_job_id]["cancel"]:
                #yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                break

            epoch_path = globals.TMP_TRAIN_CHECKPOINT_DIR / f"{req_obj.train_job_id}_epoch_{epoch}.pt"
            if req_obj.kind == "standard":                
                train_acc, train_loss = trainer.epoch(train_loader, weight=False, opt=opt)
                test_acc, test_loss = trainer.epoch(test_loader)

                if (req_obj.num_iterations == epoch) and req_obj.require_adv_attack_test:   
                    linf_adv_acc, linf_adv_loss = trainer.epoch_adv(test_loader, trainer.pgd_linf, weight=False, epsilon=trainer.eps_linf)
                    #l2_adv_acc, l2_adv_loss = trainer.epoch_adv(test_loader, trainer.pgd_l2, weight=False, epsilon=trainer.eps_l2)
                else:
                    linf_adv_acc, linf_adv_loss = -1, -1
                    #l2_adv_acc, l2_adv_loss = -1, -1

                torch.save(model.state_dict(), epoch_path)
                print(f"Saved temporary checkpoint: {epoch_path}")    

            elif req_obj.kind == "adversarial":
                attack = trainer.pgd_linf if req_obj.attack == "PGD-linf" else trainer.pgd_l2
                train_acc, train_loss = trainer.epoch_adv(train_loader, attack=attack, weight=False, opt=opt, epsilon=req_obj.epsilon, alpha = req_obj.alpha  )
                test_acc, test_loss = trainer.epoch(test_loader)

                if (req_obj.num_iterations == epoch) and req_obj.require_adv_attack_test:  
                    linf_adv_acc, linf_adv_loss = trainer.epoch_adv(test_loader, trainer.pgd_linf, weight=False, epsilon=trainer.eps_linf)

                else:
                    linf_adv_acc, linf_adv_loss = -1, -1
                    #l2_adv_acc, l2_adv_loss = -1, -1

                torch.save(model.state_dict(), epoch_path)
                print(f"Saved temporary checkpoint: {epoch_path}")    

            else:
                print("ulalalalla, no such a kind of train")
            epoch_data = {
                "type": "epoch",
                "epoch": epoch,
                "train_acc": train_acc,
                #"train_loss": train_loss,
                "test_acc": test_acc,
                #"test_loss": test_loss,
                "linf_adv_acc": linf_adv_acc,
                #"linf_adv_loss": linf_adv_loss,
                #"l2_adv_acc": l2_adv_acc,
                #"l2_adv_loss": l2_adv_loss,
            }

            all_epochs.append(epoch_data)
            #yield f"data: {json.dumps({'epoch': epoch})}\n\n"
            yield f"data: {json.dumps(epoch_data)}\n\n"

        #progress_queue.put({"type": "done"})
        if ACTIVE_JOBS["training"][req_obj.train_job_id]["cancel"]:            
            run_info["status"]="cancelled"
            
        else:
            run_info["status"]= "done"
           
            

        run_info["epochs"] = all_epochs
        run_info["timestamp"] = datetime.datetime.now().isoformat()


        if os.path.exists(globals.TRAINING_HISTORY_PATH):
            with open(globals.TRAINING_HISTORY_PATH, "r") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
        else:
            history = []

        if not isinstance(history, list):
            history = [history]

        # Append new run
        history.append(run_info)

        # Save back
        with open(globals.TRAINING_HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=4)

        print(f"Saved training result for job {run_info['train_job_id']}")

        if ACTIVE_JOBS["training"][req_obj.train_job_id]["cancel"]:            
            yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
            
        else:
            #run_info["status"]= "done"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"


        ACTIVE_JOBS["training"].pop(req.train_job_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")





@app.delete("/cancel_train/{train_id}")
def cancel_training(train_id: str):
    job = ACTIVE_JOBS["training"].get(train_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["cancel"]= True   # signal to worker
    return {"status": "cancellation requested", "id": train_id}




@app.delete("/delete_checkpoints/{trainId}")
def delete_checkpoints(trainId: str):
    folder_dir = globals.TMP_TRAIN_CHECKPOINT_DIR
    if not folder_dir.exists():
        return JSONResponse(content={"status": "ok", "message": "No checkpoint folder found."})

    deleted_files = 0
    # Delete all matching checkpoint files (trainId_epoch_*.pt)
    for file in folder_dir.glob(f"{trainId}_epoch_*.pt"):
        try:
            file.unlink()
            deleted_files += 1
        except Exception as e:
            print(f"⚠️ Could not delete {file}: {e}")

    return JSONResponse(content={
        "status": "success",
        "message": f"🧹 Deleted {deleted_files} checkpoint files for {trainId}."
    })

# --- maintain trained model ---

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
    # Copy model file
    dest_model_path = save_path / f"{model_id}.pt"
    shutil.copy2(checkpoint_path, dest_model_path)

    # Save metadata as JSON
    dest_info_path = save_path / f"{model_id}_info.json"
    with open(dest_info_path, "w", encoding="utf-8") as f:
        json.dump(info.model_dump(), f, indent=4)


    # Delete ALL temporary checkpoints for this trainId ----
    for f in folder_dir.glob(f"{trainId}_epoch_*.pt"):
        try:
            f.unlink()
        except Exception as e:
            print(f"⚠️Failed to delete {f}: {e}")

    print(f"🧹Deleted checkpoints for {trainId}")

    return JSONResponse(content={
        "status": "success",
        "message": "✅Model saved."
    })



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
                print(f"⚠️ Failed to load model info {filename}: {e}")
                continue

    return model_infos


@app.delete("/delete_model/{modelId}")
def delete_model(modelId: str):
    """
    Delete both the saved model file (.pt) and its info JSON.
    """

    folder_dir = globals.PERMANENT_TRAIN_MODELS_DIR

    if not folder_dir.exists():
        raise HTTPException(status_code=500, detail=f"Permanent model directory {folder_dir} does not exist")

    # Define file paths
    model_path = folder_dir / f"{modelId}.pt"
    info_path = folder_dir / f"{modelId}_info.json"

    deleted_files = 0
    # Delete model file
    if model_path.exists():
        try:
            model_path.unlink()
            deleted_files += 1
        except Exception as e:
            print(f"⚠️ Could not delete {model_path}: {e}")

    # Delete metadata file
    if info_path.exists():
        try:
            info_path.unlink()
            deleted_files += 1
        except Exception as e:
            print(f"⚠️ Could not delete {info_path}: {e}")

    # Handle case where neither file exists
    if deleted_files != 2:
        raise HTTPException(status_code=404, detail=f"No saved model found for ID '{modelId}'")

    return JSONResponse(content={
        "status": "success",
        "message": f"✅ Deleted model '{modelId}' and 2 related files."
    })

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
                print(f"⚠️ Could not delete {file}: {e}")

    return JSONResponse(content={
        "status": "success",
        "message": f"🧹 Deleted all model files."
    })



@app.get("/download_model/{model_id}")
def download_model(model_id: str, display_name: str | None = Query(default=None)):
    file_path = globals.PERMANENT_TRAIN_MODELS_DIR / f"{model_id}.pt"
    if not file_path.exists():
        return {"error": f"Model {model_id} not found."}

    # sanitize fallback
    filename = display_name if display_name else f"{model_id}.pt"
    return FileResponse(file_path, filename=filename, media_type="application/octet-stream")



@app.post("/evaluate_model")
def evaluate_model(req: EvaluationRequest):
    """
    Load a trained model and evaluate it directly (no worker).
    """
    model_path = globals.PERMANENT_TRAIN_MODELS_DIR / f"{req.model_id}.pt"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    try:
        acc = evaluate(req, model_path)
        return {"accuracy": acc}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")





# ------------------ Local run (for development) ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
