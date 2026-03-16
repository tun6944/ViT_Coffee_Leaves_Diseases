import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from collections import Counter
from transformers import ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime, timedelta
from glob import glob
import os, shutil, yaml, re, unicodedata
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.amp import autocast
from timm.data.mixup import Mixup
from tqdm.auto import tqdm
from tqdm.notebook import tqdm
#  INPUT datasets (giữ nguyên nguồn Roboflow) 
DATASETS = [
    "/kaggle/input/bracol-for-yolov8-detection/BRACOL_REVIEWED_ANNOTATIONS/BRACOL_REVIEWED",
    "/kaggle/input/coffee-leaves/BRACOL-VALIDADO.v11i.yolov11",
    "/kaggle/input/coffee-leaves/Coffee Leaf Diseases - OD.v10i.yolov11",
    "/kaggle/input/coffee-leaves/Coffee Leaf.v7-train-valid-test-70-20-10.yolov11",
    "/kaggle/input/coffee-leaves/Coffee leaf diseases classification.v5i.yolov11",
    "/kaggle/input/coffee-leaves/Coffee_Disease_BRACOL.v3i.yolov11",
    "/kaggle/input/coffee-leaves/Folhas.v8i.yolov11",
    "/kaggle/input/coffee-leaves/Leaf plants clasification.v13i.yolov11",
    "/kaggle/input/coffee-leaves/My First Project.v23i.yolov11",
    "/kaggle/input/coffee-leaves/My First Project.v2i.yolov11",
    "/kaggle/input/coffee-leaves/NeuralCoffe.v15i.yolov11",
    "/kaggle/input/coffee-leaves/cashew",
    "/kaggle/input/coffee-leaves/coffee leaf disease.v3i.yolov11",
    "/kaggle/input/coffee-leaves/coffee leaf.v7i.yolov11",
    "/kaggle/input/coffee-leaves/coffee leaves.v1i.yolov11",
    "/kaggle/input/coffee-leaves/coffee.v2i.yolov11",
    "/kaggle/input/coffee-leaves/datasets",
    "/kaggle/input/coffee-leaves/nhan_dien_benh_tren_la_ca_phe.v2i.yolov11"
]

#  Canonical classes (VN không dấu) 
CANONICAL_NAMES = [
    "nam_ri_sat", 
    "sau_duc_la", 
    "dom_mat_cua",
    "khoe_manh", 
    "phoma", 
    "than_thu"
]

SYNONYMS = {
    "nam_ri_sat": {"nam ri sat", "nam_ri_sat", "Rust", "coffee_rust", "leaf rust", "leaf_rust"}, #"rust"
    "sau_duc_la": {"Miner", "miner", "coffee_miner"}, #"leaf_miner", "sau_duc_la", "sau duc la"
    "dom_mat_cua": {"dom_mat_nau", "brown_eye_spot", "brown eye spot", "brown_eye", "cercospora", "cerscospora", "dom mat nau", "Cercospora", "Brown-Spot", "brown_spot", "cercosporiose", "Cercosporiose", "brown spot"},
    "khoe_manh": {"khoe_manh", "khoe manh", "healthyleaf", "Healthy-Leaf", "Healthy_Leaf", "healthy_leaf", "Healthy"}, #"healthy", "coffee_healthy"
    "phoma": {"Phoma", "phoma", "coffee_phoma"}, #
    "than_thu": {"than thu", "than_thu", "anthracnose", "Anthracnose", "Antracnose"}
}

#  Paths 
MERGED_ROOT = "merged"
ROI_ROOT    = "roi"
CKPT_ROOT   = "checkpoints"

#  YOLO train settings 
YOLO_MODEL = "yolo11s.pt"   # cập nhật theo yêu cầu
IMGSZ_YOLO = 640
EPOCHS_YOLO = 7
DEVICE_YOLO = 'cuda'
WORKERS_YOLO = os.cpu_count() // 2

#  ROI options 
ROI_SIZE = 224

#  ViT settings 
VIT_MODEL_NAME = 'google/vit-base-patch16-224'
EPOCHS_VIT = 25
BASE_LR_VIT = 1e-5
BATCH_VIT = 16
MIXUP = 0.5
LABEL_SMOOTH = 0.1
WEIGHT_DECAY = 0.05
print(datetime.utcnow() + timedelta(hours=7))
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def norm_name(name: str) -> str:
    n = strip_accents(name.strip())
    n = re.sub(r"[^a-zA-Z0-9\s\-]+", "_", n).strip('_')
    print(n)
    return n

def canonicalize(name: str):
    n = norm_name(name)
    for cano, pool in SYNONYMS.items():
        if n in pool or n == cano:
            return cano
    return None
"""
for split in ["train", "valid", "test"]:
    Path(f"{MERGED_ROOT}/{split}/images").mkdir(parents=True, exist_ok=True)
    Path(f"{MERGED_ROOT}/{split}/labels").mkdir(parents=True, exist_ok=True)

SPLIT_ALIASES = {
    "train": ["train"],
    "valid": ["valid", "val"],  # chấp nhận cả 'val' và chuẩn hóa về 'valid'
    "test":  ["test"]
}

def resolve_split_dirs(ds_root: Path):
    
    #Trả về dict: split_chuan ('train'/'valid'/'test') -> (img_dir, lbl_dir)
    #Hỗ trợ 2 layout:
    #  A) Roboflow:    <root>/<split>/images, <root>/<split>/labels
    #  B) Ultralytics: <root>/images/<split>, <root>/labels/<split>
    
    results = {}

    # --- Layout A: <root>/<alias>/{images,labels} ---
    for split_std, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            img_a = ds_root / alias / "images"
            lbl_a = ds_root / alias / "labels"
            if img_a.exists() and lbl_a.exists():
                results[split_std] = (img_a, lbl_a)
                break

    # --- Layout B: <root>/{images,labels}/<split> ---
    # Chỉ bổ sung nếu chưa có split đó trong results
    for split_std, aliases in SPLIT_ALIASES.items():
        if split_std in results:
            continue
        alias_primary = aliases[0]
        img_b = ds_root / "images" / alias_primary
        lbl_b = ds_root / "labels" / alias_primary
        if img_b.exists() and lbl_b.exists():
            results[split_std] = (img_b, lbl_b)

    return results

stats = {s: defaultdict(int) for s in ["train","valid","test"]}

for ds in DATASETS:
    ds_root = Path(ds)
    if not ds_root.exists():
        print(f"[WARN] Dataset không tồn tại: {ds}")
        continue
    yaml_path = ds_root / "data.yaml"
    names = None
    if yaml_path.exists():
        with open(yaml_path,'r') as f:
            meta = yaml.safe_load(f)
        names = meta.get('names')
        if isinstance(names, dict):
            try:
                names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
            except Exception:
                # fallback: sắp theo khóa như chuỗi
                names = [names[k] for k in sorted(names.keys())]
    id_map = {}
    if names is not None:
        for old_id, name in enumerate(names):
            cano = canonicalize(str(name))
            if cano is None:
                print(f"[WARN] '{name}' không map -> bỏ bbox lớp này.")
                continue
            id_map[old_id] = CANONICAL_NAMES.index(cano)
    prefix = ds_root.name.replace(' ','_')    
    # Phát hiện cấu trúc split và lặp
    split_dirs = resolve_split_dirs(ds_root)

    for split_std, (img_dir, lbl_dir) in split_dirs.items():
        # copy images
        for img in img_dir.iterdir():
            if img.is_file() and img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                shutil.copy2(img, Path(f"{MERGED_ROOT}/{split_std}/images")/f"{prefix}_{img.name}")
                stats[split_std]['images'] += 1

        # remap labels
        for txt in lbl_dir.glob('*.txt'):
            lines_out = []
            with open(txt,'r') as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln: 
                        continue
                    parts = ln.split()
                    old_id = int(parts[0])
                    if old_id not in id_map:
                        stats[split_std]['labels_dropped'] += 1
                        continue
                    parts[0] = str(id_map[old_id])
                    lines_out.append(' '.join(parts))
            if lines_out:
                out = Path(f"{MERGED_ROOT}/{split_std}/labels")/f"{prefix}_{txt.name}"
                with open(out,'w') as g:
                    g.write('\n'.join(lines_out)+'\n')
                stats[split_std]['labels_kept'] += 1

merged_yaml = {
    'path': MERGED_ROOT,
    'train': 'train/images',
    'val':   'valid/images',
    'test':  'test/images',
    'names': {i:n for i,n in enumerate(CANONICAL_NAMES)}
}
#with open(Path(MERGED_ROOT)/'merged_data.yaml','w',encoding='utf-8') as f:
with open('merged_data.yaml','w',encoding='utf-8') as f:
    yaml.safe_dump(merged_yaml, f, sort_keys=False, allow_unicode=True)
print('MERGE DONE')
for s in ["train","valid","test"]:
    print(s, dict(stats[s]))

print(datetime.utcnow() + timedelta(hours=7))
for split in ['train','valid','test']:
    for cname in CANONICAL_NAMES:
        Path(f"{ROI_ROOT}/{split}/{cname}").mkdir(parents=True, exist_ok=True)

def union_boxes(boxes, w, h, margin=0.10):
    if len(boxes)==0: return None
    x1 = max(0, int(np.min(boxes[:,0]))); y1 = max(0, int(np.min(boxes[:,1])))
    x2 = min(w, int(np.max(boxes[:,2]))); y2 = min(h, int(np.max(boxes[:,3])))
    bw, bh = x2-x1, y2-y1
    x1 = max(0, int(x1 - margin*bw)); y1 = max(0, int(y1 - margin*bh))
    x2 = min(w, int(x2 + margin*bw)); y2 = min(h, int(y2 + margin*bh))
    return (x1,y1,x2,y2)

def crop_square(img, box):
    x1,y1,x2,y2 = box
    roi = img.crop((x1,y1,x2,y2))
    side = max(roi.size)
    dw = side - roi.size[0]; dh = side - roi.size[1]
    padding = (dw//2, dh//2, dw - dw//2, dh - dh//2)
    return ImageOps.expand(roi, padding, fill=(0,0,0))

def boxes_from_gt(label_path):
    items=[]
    if not label_path.exists(): 
        return items
    with open(label_path,'r') as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 5: 
                continue
            cls = int(parts[0]); xc,yc,w,h = map(float, parts[1:5])
            items.append((cls, xc, yc, w, h))
    return items

def yolo_xyxy_from_norm(xc,yc,w,h,W,H):
    return int(xc*W - w*W/2), int(yc*H - h*H/2), int(xc*W + w*W/2), int(yc*H + h*H/2)

def majority_class_gt(gt_items):
    if not gt_items: return None
    ids = [int(g[0]) for g in gt_items]
    return Counter(ids).most_common(1)[0][0]

for split in ['train','valid','test']:
    src_images = Path(f"{MERGED_ROOT}/{split}/images")
    src_labels = Path(f"{MERGED_ROOT}/{split}/labels")
    total=roi_saved=skipped=0
    for img_path in src_images.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in {'.jpg','.jpeg','.png'}:
            continue
        total += 1
        lab_path = src_labels / (img_path.stem + '.txt')
        gts = boxes_from_gt(lab_path)
        if not gts:
            skipped += 1
            continue

        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        # crop từng bbox
        for i, (cls, xc,yc,w,h) in enumerate(gts):
            x1,y1,x2,y2 = yolo_xyxy_from_norm(xc,yc,w,h,W,H)
            # clamp biên ảnh
            x1 = max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
            if x2<=x1 or y2<=y1:
                continue
            roi = img.crop((x1,y1,x2,y2)).resize((ROI_SIZE, ROI_SIZE))
            # map id -> tên lớp
            if 0 <= cls < len(CANONICAL_NAMES):
                cname = CANONICAL_NAMES[cls]
            else:
                # id lạ thì bỏ qua crop này (đảm bảo nhãn đúng)
                continue
            out_name = f"{img_path.stem}_{i}{img_path.suffix}"
            out_p = Path(f"{ROI_ROOT}/{split}/{cname}/{out_name}")
            roi.save(out_p)
            roi_saved += 1

    print(f"[ROI {split}] total={total}, roi_saved={roi_saved}, skipped(no GT)={skipped}")
print('ROI dataset:', ROI_ROOT)

if os.path.exists(MERGED_ROOT):
    shutil.rmtree(MERGED_ROOT)
    print(f"Đã xóa thư mục{MERGED_ROOT}")
"""
def count_per_class(root):
    counts = {}
    for c in CANONICAL_NAMES:
        d = Path(root)/c
        counts[c] = sum(1 for p in d.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
    return counts

print("ROI train counts:", count_per_class(f"{ROI_ROOT}/train"))
print("ROI valid counts:", count_per_class(f"{ROI_ROOT}/valid"))
print("ROI test counts:",  count_per_class(f"{ROI_ROOT}/test"))

print(datetime.utcnow() + timedelta(hours=7))