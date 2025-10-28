# Compresso

**Compresso** is an interactive application for **dataset compression** based on the concept of **Minimal Finite Covering (MFC)**.  
It provides an intuitive, visual tool for exploring how datasets can be compressed while preserving their essential structure and information.



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
git clone https://github.com/blabla  (waiting for it temporarily)
```
```
cd Compresso
```

#### 2. Prepare the Gurobi license

The project includes three optimizers (CBC,SCIP,Gurobi) for linear optimization, among which **Gurobi** is the most effective and recommended.  
It requires a license for personal or academic use. You can apply for one at [https://www.gurobi.com](https://www.gurobi.com) under **Gurobi WLS (Web License Service)**.

If you already have a local license, note that running inside Docker requires a **WLS license**.  
If you already use a WLS license for another container, you can reuse it for this application.  

Below is the setup process. You only need to do this once — your license credentials can be reused for future runs unless the license expires or is revoked.

In the project’s root folder, create a file named `.env` (if it doesn’t exist yet), and paste your credentials from the license:

```
GRB_WLSACCESSID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
GRB_WLSSECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GRB_LICENSEID=1234567
```

The `docker-compose.yml` file already includes the environment section to export your credentials automatically, so you don’t need to modify it:

```
environment:
  - GRB_WLSACCESSID=${GRB_WLSACCESSID}
  - GRB_WLSSECRET=${GRB_WLSSECRET}
  - GRB_LICENSEID=${GRB_LICENSEID}
```

#### 3. Build and run the application
```
docker compose up --build
```

#### 4. Open Compresso in your browser  
http://localhost:3000

#### 5. Stop the application
```
docker compose down
```
or use **Ctrl + C** in the terminal.



## How It Works:

The backend automatically downloads and preprocesses the four built-in datasets on startup.

All permanent and temporary data is stored under `data/`, which is accessible to the user.  
The `data/` directory inside the container mirrors your local folder `CompressoApp/data/`, allowing users to view and manage their data directly.

The frontend communicates with the backend API to visualize compressed datasets, perform training, and interactively explore MFC-based compression results.



## Common Issues and Solutions:

**Problem:** Slow first startup  
**Cause:** Dataset downloading and preprocessing  
**Solution:** Wait until “webmcs-frontend  | info: Microsoft.Hosting.Lifetime[0]” appears in the logs

More issues may be added as they arise.



## Note  
- Modify `docker-compose.yml` to change ports if needed  
- Rebuild containers after code changes:  
  ```
  docker compose up --build
  ```



## License:
This project is released under the **MIT License**.  
You are free to use, modify, and distribute it with attribution.



## Author:
**Ying Pei**  
pwb749@alumni.ku.dk
