from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import numpy as np
import io
from ultralytics import YOLO
from transformers import ViTForImageClassification, ViTImageProcessor
from fastapi.responses import HTMLResponse
#  CONFIG 
YOLO_MODEL_PATH = "models/best.pt"
VIT_MODEL_DIR = "models/vit_best"
DEVICE = "cuda"
CONF_THRES = 0.05
IMG_SIZE = 224
#

app = FastAPI(title="Coffee Leaf Disease Detection API")

# STATIC FILES
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#  LOAD MODELS 
yolo = YOLO(YOLO_MODEL_PATH).to(DEVICE)

vit = ViTForImageClassification.from_pretrained(VIT_MODEL_DIR).to(DEVICE)
vit.eval()

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
id2label = vit.config.id2label

@app.get("/", response_class=HTMLResponse)
def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()
@app.get("/health")
def health():
    return {"status": "ok"}

def crop_image(img, box):
    x1, y1, x2, y2 = map(int, box)
    return img[y1:y2, x1:x2]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)

    results = yolo(img_np, conf=CONF_THRES, max_det=5, verbose=False)[0]
    detections = []

    # Log số lượng bbox YOLO detect được
    if results.boxes is None or len(results.boxes) == 0:
        print("[YOLO] No boxes detected.")
        return {"num_detections": 0, "detections": []}
    else:
        print(f"[YOLO] Number of boxes detected: {len(results.boxes)}")

    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        det_conf = float(box.conf[0])
        print(f"[YOLO] Box {idx}: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), conf={det_conf:.3f}")

        roi = crop_image(img_np, (x1, y1, x2, y2))
        if roi.size == 0:
            print(f"[WARN] ROI {idx} is empty, skip.")
            continue

        roi_pil = Image.fromarray(roi).resize((IMG_SIZE, IMG_SIZE))
        inputs = processor(images=roi_pil, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            out = vit(**inputs)
            probs = torch.softmax(out.logits, dim=1)
            score, pred = torch.max(probs, dim=1)

        print(f"[ViT] ROI {idx}: pred={int(pred)}, class={id2label[int(pred)]}, score={float(score):.4f}")

        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "class_id": int(pred),
            "class_name": id2label[int(pred)],
            "confidence": round(float(score), 4),
            "det_confidence": round(det_conf, 4)
        })

    print(f"[RESULT] Total detections: {len(detections)}")
    return {
        "num_detections": len(detections),
        "detections": detections
    }
