from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the model (same architecture as before)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
model.load_state_dict(torch.load('model/cat_dog_model.pth', map_location=device))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# API endpoint for predictions
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

        # Load and preprocess the image
        image = Image.open(file.file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            class_names = ["cat", "dog"]
            prediction = class_names[predicted.item()]

        return {"prediction": prediction}

    except HTTPException as http_exc:
        # Return specific error for invalid file types
        return JSONResponse(status_code=http_exc.status_code, content={"error": http_exc.detail})

    except Exception as e:
        # Handle generic errors
        return {"error": f"An error occurred: {str(e)}"}
