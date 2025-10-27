#from urllib import request
#from fastapi.encoders import jsonable_encoder
# from datetime import datetime
# from io import BytesIO
# from typing import Optional
# from uuid import uuid4
#import networkx as nx
# from networkx.readwrite import json_graph
#from pydantic import BaseModel
#from torch.utils.data import Dataset, DataLoader
#from multiprocessing import Process, Queue, Event


from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse, Response, JSONResponse  #HTMLResponse
import plotly.io as pio
from fastapi.middleware.cors import CORSMiddleware
import json
import multiprocessing as mp
from contextlib import asynccontextmanager
#import uvicorn
import asyncio
import pickle
import time
import shutil
import os
import zipfile


import globals
from models import *
from util_train import train_worker
from util_compression import *
from util_data import *
from util_image import get_images, draw_graph, transform_images_for_frontend, tensor_to_image_bytes
from util_train import *



# ------------------ Global setup ------------------

# Limit PyTorch CPU threads for safe multi-processing
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Set multiprocessing start method safely
try:
    mp.set_start_method("spawn", force=True)  # safer for torch + macOS + Docker
except RuntimeError:
    # already set by another module
    pass


# ------------------ Unified lifespan ------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Unified lifecycle: handles startup + shutdown."""

    # --- Startup phase ---
    print("🚀 Starting backend initialization...")

    app.state.compression_jobs = {}
    app.state.training_jobs = {}
    app.state.compressed_data_dict = {}

    # Clean temp dirs
    for path in [globals.TMP_DATA_FOR_GRAPH_DIR, globals.TMP_DATA_OF_COMPRESSION_DIR,
                  globals.TMP_TRAIN_CHECKPOINT_DIR]:
        if path.exists():
            for file in path.iterdir():
                if file.is_file():
                    file.unlink()
            print(f"✅ Cleared all files in {path}")

    # Preload standard datasets
    for id in ["mnist", "cifar10", "cifar100", "svhn"]:
        # create_origin_data_obj(id, percent=10)
        # create_test_data_obj(id, percent=20)
                # delete the raw data, which is usually large and will not be used any longer
        preprocess_data_to_obj(id, train_percent=100, test_percent=100)
        #preprocess_data_to_obj(id, train_percent=10, test_percent=10)



    print("✅ All global state initialized")
    print("✅ Backend startup complete.")

    # Hand control to FastAPI runtime
    yield

    # --- Shutdown phase ---
    print("Cleaning up worker processes...")

    for job_dict_name in ("compression_jobs", "training_jobs"):
        if hasattr(app.state, job_dict_name):
            job_dict = getattr(app.state, job_dict_name)
            for jobid, job in list(job_dict.items()):
                proc = job.get("process")
                if proc and proc.is_alive():
                    print(f"  Terminating job {jobid} (pid={proc.pid})")
                    proc.terminate()
                    proc.join(timeout=2)

                # Close associated queues and events
                for key in ("queue", "results", "cancel"):
                    obj = job.get(key)
                    if obj:
                        try:
                            obj.close()
                        except Exception:
                            pass

            job_dict.clear()

    print("Cleanup finished. Server shutdown complete.")


# ------------------ FastAPI app ------------------

app = FastAPI(lifespan=lifespan)




# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     #app.state.MAX_CONCURRENT_COMPRESSION_JOBS = 5 #no use yes
#     #app.state.MAX_CONCURRENT_TRAINING_JOBS = 5 #no use yes
#     #app.state.MAX_MEMORY_DATA_NUM = 20
    
#     app.state.compression_jobs = {}
#     app.state.training_jobs = {}
#     app.state.compressed_data_dict = {}


#     for id in ["mnist", "cifar10", "cifar100", "svhn"]:
#         create_origin_data_obj(id, percent=10)
#         create_test_data_obj(id, percent=15)
        
#         # delete the raw data, which is usually large and will not be used any longer
#         folder = globals.TMP_RAW_DATA_DIR / id

#         if folder.exists() and folder.is_dir():
#             shutil.rmtree(folder)   # Recursively delete the folder and all its contents
#             print(f"🧹 Deleted dataset folder: {folder}")
#         else:
#             print(f"⚠️ Folder does not exist: {folder}")



#     print("✅All global state initialized")

#     yield

#     # --- cleanup ---
#     for attr in [
#         "compression_jobs",
#         "training_jobs",
#         "compressed_data_dict",
#     ]:
#         if hasattr(app.state, attr):
#             getattr(app.state, attr).clear()


# # ------------------ FastAPI app ------------------
# app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------ Compression  ------------------
@app.post("/compress")
async def start_compression(req: CompressRequest):
    start_time = time.time()
    
    cancel_event = multiprocessing.Event()
    progress_queue = multiprocessing.Queue()
    #result_queue = multiprocessing.Queue() 
    print("begin in start_compresion?")
    proc = multiprocessing.Process(
        target = worker_process,
        args = (req, start_time, progress_queue, cancel_event)
    )
    print("finished MFC_by_label?")

    app.state.compression_jobs[req.compression_job_id] = {
        "process": proc,
        "queue": progress_queue,
        "cancel": cancel_event
        #"results": result_queue,
    }
    proc.start()
    # launch track_compression_completion in background
    asyncio.create_task(track_compression_completion(req.compression_job_id))
    return StartCompressionResponse(success=True, message="Compression started successfully")


@app.get("/compress/{compression_job_id}")
async def stream_progress(compression_job_id: str):
    job = app.state.compression_jobs.get(compression_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    progress_queue = job["queue"]

    async def event_stream():
        while True:
            if not progress_queue.empty():
                msg = progress_queue.get()
                yield f"data: {json.dumps(msg)}\n\n".encode("utf-8")
                if msg.get("done") or msg.get("cancelled") or msg.get("error"):
                    break
            else:
                await asyncio.sleep(0.05)
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/compress/{compression_job_id}")
def cancel_compression(compression_job_id: str):
    job = app.state.compression_jobs.get(compression_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["cancel"].set()   # signal to worker
    job["queue"].put({"cancelled": True})
    return {"status": "cancellation requested", "id": compression_job_id}


async def track_compression_completion(compression_job_id: str):
    print("has enter track_compression_competion????")
    job = app.state.compression_jobs.get(compression_job_id)
    if not job:
        return
    proc = job["process"]
    #result_queue = job["results"]

    # Wait for process to finish
    while proc.is_alive():
        await asyncio.sleep(0.1)
    
    # try:
    #     # block briefly to allow result to be delivered
    #     result = result_queue.get(timeout=100)
    # except Exception:
    #     result = None

    proc.join()  # wait for completion
    # Clean up the job
    await asyncio.sleep(1)

    app.state.compression_jobs.pop(compression_job_id, None)

    save_path = f"{globals.TMP_DATA_OF_COMPRESSION_DIR}/{compression_job_id}_compressed.pt"
    
    obj = torch.load(save_path, map_location="cpu", weights_only=False)

    compressed_dataset_obj =  CompressedDatasetObj(
                                compression_id = compression_job_id, 
                                compressed_data_by_label= obj["compressed_data_by_label"],
                                summary = obj["summary"],
                                offsets_by_label = obj["offsets_by_label"],
                                )
    # if result is None:
    #     print("No result from worker; likely cancelled or crashed")
    # else:
    #     compressed_data_by_label, nodes_by_label, summary, offsets_by_label = result
    #     compressed_dataset_obj =  CompressedDatasetObj(
    #                                 compression_id = compression_job_id, 
    #                                 compressed_data_by_label= compressed_data_by_label,
    #                                 nodes_by_label = nodes_by_label,
    #                                 summary = summary, 
    #                                 offsets_by_label = offsets_by_label,
    #                                 )
    #     # Save in memory
    app.state.compressed_data_dict[compression_job_id] = compressed_dataset_obj

    # for key in app.state.compressed_data_dict.keys():
    #     print(key)
    # Optional eviction
    # if len(app.state.compressed_data_dict) > app.state.MAX_MEMORY_DATA_NUM:
    #     container_for_timestamps = {
    #             obj_id: obj.summary.timestamp
    #             for obj_id, obj in app.state.compressed_data_dict.items()
    #         }
    #     oldest = min(container_for_timestamps, key = container_for_timestamps.get)
    #     app.state.compressed_data_dict.pop(oldest, None)
    #     print("Has evicted the oldest in the memory due to memory limitaton")




# ------------------ Image supports  ------------------
@app.get("/sample_origin_images")
def sample_origin_images(dataset_name: str, label: str, n: int):
    path = f"{globals.DATA_PER_LABEL_DIR}/{dataset_name}/{label}.pt"
    obj = torch.load(path, weights_only=False)
    dataset_for_one_label = obj.stacked_tensor
    
    if dataset_for_one_label is None or dataset_for_one_label.numel() == 0:
        raise HTTPException(404, detail="Dataset not found")

    _, images = get_images(dataset_for_one_label, n, 0, random_mode = True)
    return transform_images_for_frontend(images)


@app.get("/sample_compressed_images")
def sample_compressed_images(compression_job_id:str, label: str, n: int):
    
    compressed_data_obj = app.state.compressed_data_dict[compression_job_id]
    dataset_by_label = compressed_data_obj.compressed_data_by_label
    sequential_offsets = compressed_data_obj.offsets_by_label
    labels = compressed_data_obj.summary.labels
    if label not in labels:
        raise HTTPException(404, detail=f"Label {label} not found")
    sequential_offset = sequential_offsets[label]
    new_sequential_offset, images = get_images(
             dataset_by_label[label], n, sequential_offset, random_mode = False)
    app.state.compressed_data_dict[compression_job_id].offsets_by_label[label] = new_sequential_offset
    return transform_images_for_frontend(images)


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
    #data_obj = app.state.compressed_data_dict[compression_job_id]
    read_path = f"{globals.TMP_DATA_FOR_GRAPH_DIR}/{label}_{compression_job_id}_nodes.pt"
    nodes =torch.load(read_path, map_location="cpu", weights_only=False)

    #nodes = data_obj.nodes_by_label[label]
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
            print(f"⚠️ Could not delete {file}: {e}")
    return JSONResponse(content={
        "status": "success",
        "message": f"🧹 Deleted {deleted_files} graph files for compressionId '{compressionId}'."
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

    # 🧹 Delete all files in the directory
    for file in folder_dir.iterdir():
        if file.is_file():
            try:
                file.unlink()
                deleted_files += 1
            except Exception as e:
                print(f"⚠️ Could not delete {file}: {e}")

    return JSONResponse(content={
        "status": "success",
        "message": f"🧹 Deleted {deleted_files} graph files."
    })


# ------------------ Summary supports  ------------------

@app.get("/fetch_compression_summary_from_memory/{compression_job_id}")
async def fetch_compression_summary_from_memory(compression_job_id: str):
    container = app.state.compressed_data_dict
    while compression_job_id not in container:
        await asyncio.sleep(0.1)  # adjust sleep interval as needed
    # Wait for the summary to be populated
    while getattr(container[compression_job_id], "summary", None) is None:
        await asyncio.sleep(0.1)
    summary = getattr(container[compression_job_id], "summary")
    return {"summary": summary, "status": "done"}



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
                    print(f"⚠️ Failed to delete {file}: {e}")

        return JSONResponse(content={
            "status": "success",
            "message": f"Has deleted {deleted_files} files"
        })

    except Exception as e:
        print(f"❌ Error clearing container data: {e}")
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
    
    compressed_obj = app.state.compressed_data_dict.get(compression_job_id)
    current_summary = compressed_obj.summary
    if not compressed_obj:
        raise HTTPException(status_code=404, detail="Compressed data not found")
    
    dataset_name = compressed_obj.summary.dataset_name
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

    # Convert compressed dict to trainable data_obj
    data_obj = prepare_trainable_data_obj(compressed_obj.compressed_data_by_label, dataset_name = dataset_name)

    # Save .pt and summary
    torch.save(data_obj, os.path.join(file_dir, f"{compression_job_id}_compressed_data.pt"))
    with open(os.path.join(file_dir, f"{compression_job_id}_summary.json"), "w") as f:
        f.write(compressed_obj.summary.json())

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

    compressed_obj = app.state.compressed_data_dict.get(compression_job_id)
    if not compressed_obj:
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
        dataset_name = compressed_obj.summary.dataset_name
        data_obj = prepare_trainable_data_obj(compressed_obj.compressed_data_by_label, dataset_name=dataset_name)

        torch.save(data_obj, file_dir / f"{compression_job_id}_compressed_data.pt")
        with open(file_dir / f"{compression_job_id}_summary.json", "w") as f:
            f.write(compressed_obj.summary.json())

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



# ------------------ Dataset supports  ------------------

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
    path_to_delete = os.path.join(globals.DATA_PER_LABEL_DIR, dataset_name)

    #deleted_paths = []
    #skipped_paths = []

    # --- Attempt deletion for each path ---
    #for path in paths_to_delete:
    success_delete_data = False
    if os.path.exists(path_to_delete):
        try:
            shutil.rmtree(path_to_delete)
            #deleted_paths.append(path)
            success_delete_data = True
            print(f"Deleted directory: {path_to_delete}")
        except Exception as e:
            print(f"Failed to delete {path_to_delete}: {e}")
                #skipped_paths.append({"path": path, "error": str(e)})
        #else:
            #skipped_paths.append({"path": path, "error": "Not found"})
    
    ### we have decided to keep the test data, even though the original dataset is to be
    ### deleted, because the remained compressed data may be used for training. In the case
    ### test data is useful.
    # --- Try deleting the test .pt file ---
    # test_file_path = globals.TEST_DATA_DIR / f"{dataset_name}_test.pt"
    # if os.path.exists(test_file_path):
    #     try:
    #         os.remove(test_file_path)
    #         deleted_paths.append(test_file_path)
    #         print(f"🧹 Deleted test file: {test_file_path}")
    #     except Exception as e:
    #         print(f"⚠️ Failed to delete test file {test_file_path}: {e}")
    #         skipped_paths.append({"path": test_file_path, "error": str(e)})
    # else:
    #     skipped_paths.append({"path": test_file_path, "error": "Not found"})

    
    #success_delete_data =  len(skipped_paths)==0 

    success = success_deregistry and success_delete_data
    if success:
        return {"message": f"✅Deleted '{dataset_name}'" }
    else:
        return {"message": f"{dataset_name} cannot be deleted completely, check it" }


# @app.post("/preprocess_user_dataset/{dataset_name}")
# def preprocess_user_dataset(dataset_name: str):

#     print(f"has enter preprocess_user_dataset, everything should start from her. {dataset_name}")
#     dataset_dir = globals.USER_RAW_DATA_DIR / dataset_name
#     print("Current working directory when just register:", os.getcwd())
#     print(dataset_dir)

#     #dataset_dir = globals.USER_RAW_DATA_DIR / dataset_name


#     #print(dataset_dir)
#     if not os.path.exists(dataset_dir):
#         raise HTTPException(404, detail=f"Dataset '{dataset_name}' not found")

#     try:
#         create_origin_data_obj(dataset_name)
#         create_test_data_obj(dataset_name)
#         return {
#             "status": "success",
#             "message": f"✅ Dataset '{dataset_name}' preprocessed successfully"
#         }
#     except Exception as e:
#         raise HTTPException(500, detail=f"Preprocessing failed: {str(e)}")




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

        # Remove existing folder if it exists
        #if os.path.exists(dataset_folder):
        #    shutil.rmtree(dataset_folder)
        #os.makedirs(dataset_folder, exist_ok=True)

        # Extract ZIP into the target directory
        with zipfile.ZipFile(temp_path, "r") as zip_ref:
            zip_ref.extractall(dataset_folder)

        # Cleanup the temp file
        os.remove(temp_path)

        print(f"Extracted dataset: {dataset_folder}")

        # return {
        #     "message": f"Uploaded and extracted {file.filename} successfully.",
        #     "dataset_folder": dataset_folder
        # }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.")
    except Exception as e:
        print(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


    try:
        preprocess_data_to_obj(dataset_name=dataset_name, train_percent=100, test_percent=100)
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
    print("helllo111")
    if not os.path.exists(history_path):
        return JSONResponse(content={
            "status": "ok",
            "message": f"No training history found at {history_path}."
        })
    print("helllo122222")
    try:
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)
        print("helllo188888")
        return JSONResponse(content={
            "status": "success",
            "message": "🧹 Training history file cleared successfully."
        })

    except Exception as e:
        print(f"⚠️ Failed to clear history file: {e}")
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

    cancel_event = multiprocessing.Event()
    progress_queue = multiprocessing.Queue()

    req_ = req.model_dump()
    print("øv, a train worker now")
    kind = req_.get("kind", "").lower()
    if kind == "standard":
        req_obj = StandardTrainRequest(**req_)
    else:
        req_obj = AdvTrainRequest(**req_)

    train_dataset = None
    compressed_obj = app.state.compressed_data_dict.get(req_obj.data_info.get("data_id"))
    if compressed_obj:
        dataset_name = compressed_obj.summary.dataset_name
        train_dataset = prepare_trainable_data_obj(compressed_obj.compressed_data_by_label, dataset_name = dataset_name)
    else:
        train_dataset_file = globals.COMPRESSION_CONTAINER_DIR / f"{req_obj.data_info.get('data_id')}_compressed_data.pt"
        train_dataset = torch.load(train_dataset_file, weights_only=False)

    async def event_stream():
        # Start training after SSE is ready
        proc = multiprocessing.Process(
                                    target=train_worker,
                                    args=(req_obj, train_dataset,
                                        progress_queue, cancel_event)
                                    )
        proc.start()

        app.state.training_jobs[req.train_job_id] = {
            "process": proc,
            "queue": progress_queue,
            "cancel": cancel_event,
        }

        while True:
            if not progress_queue.empty():
                msg = progress_queue.get()
                yield f"data: {json.dumps(msg)}\n\n".encode("utf-8")
                if msg.get("type") in ("done", "cancelled", "error"):
                    break
            else:
                await asyncio.sleep(0.05)

        progress_queue.close()
        asyncio.create_task(track_training_completion(req.train_job_id))
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/train/{train_id}")
def cancel_training(train_id: str):
    job = app.state.training_jobs.get(train_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["cancel"].set()   # signal to worker
    job["queue"].put({"cancelled": True})
    return {"status": "cancellation requested", "id": train_id}



async def track_training_completion(train_id: str):
    """
    Waits for process to finish, cleans up app state.
    Stores final model and per-epoch metrics.
    """
    job = app.state.training_jobs.get(train_id)
    if not job:
        return

    proc = job["process"]
    proc.join()  # wait for completion

    # Clean up the job
    await asyncio.sleep(0.1)
    app.state.training_jobs.pop(train_id, None)


# ------------------ Maintain saved models ------------------

# @app.post("/save_model/{trainId}/{epoch}")
# def save_model(trainId: str, epoch: int):
#     folder_dir = globals.TMP_TRAIN_CHECKPOINT_DIR
#     if not folder_dir.exists():
#         raise HTTPException(status_code=500, detail=f"Temporary checkpoint directory {folder_dir} does not exist")

#     checkpoint_path = folder_dir / f"{trainId}_epoch_{epoch}.pt"
#     if not checkpoint_path.exists():
#         raise HTTPException(status_code=404, detail=f"Checkpoint for trainId '{trainId}' epoch '{epoch}' not found")

#     save_path = globals.PERMANENT_TRAIN_MODELS_DIR
#     save_path.mkdir(parents=True, exist_ok=True)

#     # Create a destination file name
#     dest_path = save_path / f"{trainId}.pt"

#     # Copy the checkpoint file (instead of loading it)
#     shutil.copy2(checkpoint_path, dest_path)

#     return JSONResponse(content={
#         "status": "success",
#         "message": f"✅Model saved."
#     })


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
    #deleted_files = []
    for f in folder_dir.glob(f"{trainId}_epoch_*.pt"):
        try:
            f.unlink()
        except Exception as e:
            print(f"⚠️ Failed to delete {f}: {e}")

    print(f"🧹 Deleted checkpoints for {trainId}")

    return JSONResponse(content={
        "status": "success",
        "message": "✅Model saved."
    })


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

    # Also delete any subdirectory named after the trainId
    # train_dir = folder_dir / trainId
    # if train_dir.exists() and train_dir.is_dir():
    #     try:
    #         shutil.rmtree(train_dir)
    #         print(f"🧹 Deleted directory {train_dir}")
    #     except Exception as e:
    #         print(f"⚠️ Could not delete directory {train_dir}: {e}")

    return JSONResponse(content={
        "status": "success",
        "message": f"🧹 Deleted {deleted_files} checkpoint files for {trainId}."
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

# ------------------ Evaluation  ------------------
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from pathlib import Path





# --- main endpoint ---
@app.post("/evaluate_model")
def evaluate_model(req: EvaluationRequest):
    """
    Load a trained model and evaluate it directly (no worker).
    """
    model_path = globals.PERMANENT_TRAIN_MODELS_DIR / f"{req.model_id}.pt"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    try:
        # 1️⃣ Load dataset
        dataset = load_dataset(req.dataset_name, train_=req.train_)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        # 2️⃣ Create model & load weights
        sample, _ = dataset[0]
        in_channels = sample.shape[0]
        num_classes = len(load_dataset_classes()[req.dataset_name])

        model = ConvNet(in_channels=in_channels, num_classes=num_classes)

        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        # 3️⃣ Evaluate
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in loader:
                outputs = model(x)
                loss = criterion(outputs, y)
                total_loss += loss.item() * x.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / total
        acc = correct / total

        print(f"[Evaluation] {req.dataset_name} | Acc: {acc:.4f}, Loss: {avg_loss:.4f}")

        return {
            #"status": "success",
            #"dataset": req.dataset_name,
            #"split": req.train_,
            #"model_path": str(model_path),
            "accuracy": acc,
            #"avg_loss": avg_loss
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")







# # ------------------ Safe main entry ------------------
# if __name__ == "__main__":

#     # Empty the file at startup 
#     G_path = globals.TMP_DATA_FOR_GRAPH_DIR
#     T_path = globals.TMP_TRAIN_CHECKPOINT_DIR
#     for file in G_path.iterdir():
#         if file.is_file():
#             file.unlink()  # removes the file
#     for file in T_path.iterdir():
#         if file.is_file():
#             file.unlink()  # removes the file
            
#     print(f"✅Cleared all files in {G_path}")
#     print(f"✅Cleared all files in {T_path}")

    
#     import atexit

#     # --- Limit PyTorch CPU threads for safe multi-processing ---
#     os.environ["MKL_NUM_THREADS"] = "1"
#     os.environ["OMP_NUM_THREADS"] = "1"

#     def cleanup():
#         print("🧹 Cleaning up worker processes...")
#         if hasattr(app.state, "jobs"):
#             for jobid, job in list(app.state.jobs.items()):
#                 proc = job.get("process")
#                 if proc and proc.is_alive():
#                     print(f"  Terminating job {jobid} (pid={proc.pid})")
#                     proc.terminate()
#                     proc.join(timeout=2)

#                 # close queues if possible
#                 q = job.get("queue")
#                 if q:
#                     try:
#                         q.close()
#                     except Exception:
#                         pass
#                 r = job.get("results")
#                 if r:
#                     try:
#                         r.close()
#                     except Exception:
#                         pass

#             app.state.jobs.clear()
#         print("✅Cleanup finished.")

#     atexit.register(cleanup)

#     import multiprocessing as mp
#     mp.set_start_method("spawn", force=True)  # safer for torch + macOS

#     # Start your server
#     import uvicorn
#     uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
#     #uvicorn.run("api:app", host="127.0.0.1", port=8000)
    





# # ------------------ Global setup ------------------


# # --- Limit PyTorch CPU threads for safe multi-processing ---
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

# # --- Set multiprocessing start method safely ---
# try:
#     mp.set_start_method("spawn", force=True)  # safer for torch + macOS + Docker
# except RuntimeError:
#     # already set by another module
#     pass


# # ------------------ Lifecycle events ------------------

# @app.on_event("startup")
# async def startup_event():
#     """Runs once when the FastAPI app starts (both locally and in Docker)."""
#     G_path = globals.TMP_DATA_FOR_GRAPH_DIR
#     T_path = globals.TMP_TRAIN_CHECKPOINT_DIR

#     # Clear temp directories
#     for file in G_path.iterdir():
#         if file.is_file():
#             file.unlink()
#     for file in T_path.iterdir():
#         if file.is_file():
#             file.unlink()

#     print(f"✅ Cleared all files in {G_path}")
#     print(f"✅ Cleared all files in {T_path}")
#     print("✅ Backend startup complete.")


# @app.on_event("shutdown")
# async def shutdown_event():
#     """Runs when the app or container shuts down."""
#     print("🧹 Cleaning up worker processes...")

#     # Combine your compression + training jobs cleanup
#     for job_dict_name in ("compression_jobs", "training_jobs"):
#         if hasattr(app.state, job_dict_name):
#             job_dict = getattr(app.state, job_dict_name)
#             for jobid, job in list(job_dict.items()):
#                 proc = job.get("process")
#                 if proc and proc.is_alive():
#                     print(f"  Terminating job {jobid} (pid={proc.pid})")
#                     proc.terminate()
#                     proc.join(timeout=2)

#                 for key in ("queue", "results", "cancel"):
#                     obj = job.get(key)
#                     if obj:
#                         try:
#                             obj.close()
#                         except Exception:
#                             pass

#             job_dict.clear()

#     print("✅ Cleanup finished.")



# ------------------ Local run (for development) ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
