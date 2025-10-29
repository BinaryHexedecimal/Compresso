# Compresso

**Compresso** is an interactive application for **dataset compression** based on the concept of **Minimal Finite Covering (MFC)**.  It provides an intuitive, visual tool for exploring how datasets can be compressed while preserving their essential structure and information.



## Features

- **Minimal Finite Covering–based compression** for efficient data representation  
- **Interactive visualization** showing how compressed samples can represent the original dataset  
- **Multiple dataset support**, including built-in options:  
  - `MNIST`, `CIFAR-10`, `CIFAR-100`, and `SVHN`  
  - plus the ability for users to **upload custom datasets**, strictly following the required formats  
- **Integrated frontend + backend architecture** — runs locally via Docker, similar to a self-contained Jupyter-style environment  
- **Browser-based interactive interface**, making experimentation simple and visual  



## Tech Stack

- **Backend:** Python · FastAPI · PyTorch  
- **Frontend:** ASP.NET Core (C# · Blazor)  
- **Containerization:** Docker · Docker Compose  
- **Visualization:** Web-based interactive UI  



## Project Structure:
```
Compresso/
├── Backend/ (FastAPI backend service)
│   ├── data/
│   ├── requirements.txt
│   ├── xxx.py
│   └── Dockerfile
├── CompressoApp/ (ASP.NET Core frontend)
│   ├── Components/
│   ├── Models/
│   ├── Services/
│   ├── wwwroot/
│   ├── xxx.cs
│   └── Dockerfile
├── docker-compose.yml (Orchestrates backend and frontend)
└── README.md
```



## Prerequisites:

- Docker installed (https://docs.docker.com/get-docker/)  
- Docker Compose installed (usually included with Docker)



## Quick Start:

#### 1. Clone the repository
```
git clone https://github.com/BinaryHexedecimal/Compresso.git
```
```
cd Compresso
```

#### 2. Prepare the Gurobi license (Optional)

##### 2.1 Linear Optimization Setup

This project includes three linear optimization solvers: **CBC**, **SCIP**, and **Gurobi**.  
Among them, **Gurobi** generally offers the best performance and is the recommended option. However, it requires a valid license (free for personal or academic use).

##### 2.2 Choosing an Optimizer
If you prefer **not** to use Gurobi for any reason, you can skip its setup — the compression page will still provide access to the other two optimizers (**CBC** and **SCIP**).

If you wish to use **Gurobi**, you can obtain a free **Web License Service (WLS)** license at:  
[https://www.gurobi.com](https://www.gurobi.com)

> **Note:**  
> - A *local* Gurobi license (the standard `.lic` file) typically will **not** work inside Docker. You must use a **WLS (Web License Service)** license for containerized environments.  
> - If you already have a WLS license for another container, you can reuse it here.


##### 2.3 One-Time Setup

Once you have your WLS credentials, create a file named `.env` in the project’s **root directory** (`Compresso/`) and add the following environment variables:

```
GRB_WLSACCESSID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
GRB_WLSSECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GRB_LICENSEID=1234567
```

You only need to do this once — the same credentials can be reused for future runs, unless the license expires or is revoked.


#### 3. Build and run the application
```
docker compose up --build
```
> **Note:**  
> -  After the initialization, you can start existing containers simply by running:```docker compose up```
> - However, if you have made changes to the code, dependencies, or Docker configuration, you should rebuild the images using ```docker compose up --build```


#### 4. Open Compresso in your browser  
http://localhost:3000

#### 5. Stop the application
```
docker compose down
```
or use ***Ctrl + C*** in the terminal.




## How It Works:

The backend automatically downloads and preprocesses the four built-in datasets on startup.

All permanent and temporary data is stored under `data/`, which is accessible to the user.  
The `data/` directory inside the container mirrors your local folder `Compresso/Backend/data/`, allowing users to view and manage their data directly.

The frontend communicates with the backend API to visualize compressed datasets, perform training, and interactively explore MFC-based compression results.



## Common Issues and Solutions:

**Problem:** Slow first startup  

**Cause:** Installing packages and Preloading built-in datasets

**Solution:** Wait until “webmcs-frontend  | info: Microsoft.Hosting.Lifetime[0]” appears in the logs.

By default, the backend is allowed up to 40 minutes to initialize before it stops automatically. You can adjust this duration in the `docker-compose.yml` under the ***healthcheck*** section.

More issues may be added as they arise.



## Note  
- Modify `docker-compose.yml` to change ports if needed  
- For visualization purposes,  we use 15% of each of the four built-in training datasets in compression. This configuration is designed to ensure smooth operation on a typical personal computer. The parameter can be adjusted by modifying ***BUILT_IN_DATASET_PERCENT*** in `Compresso/Backend/globals.py` as needed.  



## License:
This project is released under the **MIT License**.  
You are free to use, modify, and distribute it with attribution.



## Author:
**Ying Pei**  
pwb749@alumni.ku.dk
