import os
import urllib.request
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import json

app = FastAPI(title="DenseNet-121 ONNX Inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static dir exists
os.makedirs("static_dense", exist_ok=True)
app.mount("/static", StaticFiles(directory="static_dense"), name="static")

MODEL_PATH = "densenet121-densenet-121-float/model.onnx"
CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
CLASSES_FILE = "imagenet_classes_dense.txt"

# Load ONNX model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Load ImageNet classes
if not os.path.exists(CLASSES_FILE):
    urllib.request.urlretrieve(CLASSES_URL, CLASSES_FILE)

with open(CLASSES_FILE, "r") as f:
    categories = [s.strip() for s in f.readlines()]

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    # Resize to 256
    image = image.resize((256, 256), Image.BILINEAR)
    # Center crop 224
    width, height = image.size
    new_width, new_height = 224, 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    image = image.crop((left, top, right, bottom))
    
    # To numpy array and normalize
    img_data = np.array(image).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # Transpose to [C, H, W]
    img_data = np.transpose(img_data, (2, 0, 1))
    # Add batch dimension [1, C, H, W]
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

@app.get("/")
def read_root():
    return FileResponse("static_dense/index_dense.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        input_data = preprocess_image(image)
        outputs = session.run(None, {input_name: input_data})
        logits = outputs[0]
        
        probabilities = softmax(logits)[0]
        top5_prob = np.argsort(probabilities)[-5:][::-1]
        
        results = []
        for i in top5_prob:
            results.append({
                "class": categories[i],
                "confidence": float(probabilities[i])
            })
            
        return {"success": True, "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
