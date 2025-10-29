# ======================
# Base image
# ======================
FROM python:3.11.5-slim


# Set working directory inside the container
WORKDIR /app

# ======================
# Install dependencies
# ======================
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*
    
# ======================
# Copy your code
# ======================
COPY . .

# ======================
# Environment setup
# ======================
# Ensures logs are flushed immediately
ENV PYTHONUNBUFFERED=1

# Limit threading to avoid CPU oversubscription (especially with PyTorch)
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1


# Expose FastAPI port
EXPOSE 8000

# ======================
# Start command
# ======================
# Run the app using uvicorn directly
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
