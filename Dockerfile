
# Use a Python base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy files to the container
COPY app.py ./ 
COPY model ./model
COPY requirements.txt ./ 

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    build-essential \
    libopenblas-dev

RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
