
# Use a Python base image
FROM python:3.11-slim AS builder

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    build-essential \
    libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy files to the container
COPY app.py ./ 
COPY model ./model
COPY static ./static
COPY requirements.txt ./ 
    
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8000

# Command to run the app
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]