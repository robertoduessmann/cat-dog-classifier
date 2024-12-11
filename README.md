# Cat-Dog Classifier Project

This project implements an end-to-end pipeline for building, training, and deploying a **Cat-Dog Classifier** using PyTorch and FastAPI. The classifier is based on a simple convolutional neural network (CNN) trained on the CIFAR-10 dataset.

---

## Features
- **Data Preparation**: Script to download and preprocess the CIFAR-10 dataset.
- **Model Training**: Train a CNN model to classify images into one of 10 categories.
- **Model Deployment**: Deploy the trained model as a REST API using FastAPI.
- **Docker Support**: Dockerfile provided for easy deployment.

---

## Project Structure
```
cat_dog_classifier/
├── data/                # Directory for storing dataset
├── model/               # Directory for saving the trained model
├── scripts/             # Python scripts for data preparation and training
│   ├── data_prep.py     # Script to download and preprocess the dataset
│   ├── train_model.py   # Script to train the CNN model
├── app.py               # FastAPI app for model inference
├── requirements.txt     # Dependencies required for the project
├── Dockerfile           # Dockerfile for deploying the API
```

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Pip
- PyTorch
- FastAPI
- Docker (optional, for containerization)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd cat_dog_classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Steps to Run the Project

### 1. Data Preparation
Run the script to download and preprocess the CIFAR-10 dataset:
```bash
python scripts/data_prep.py
```

### 2. Model Training
Train the CNN model:
```bash
python scripts/train_model.py
```
The trained model will be saved in the `model/` directory as `cat_dog_model.pth`.

### 3. Start the API
Run the FastAPI app:
```bash
uvicorn app:app --reload
```
Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Make Predictions
Use the `/predict` endpoint to classify an image. Example using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/predict/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path_to_image.jpg"
```

---

## Docker Deployment
1. Build the Docker image:
   ```bash
   docker build -t cat-dog-api .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 cat-dog-api
   ```

3. Access the API at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

