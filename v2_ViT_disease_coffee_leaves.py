## 1) Cài đặt & Kiểm tra môi trường

# For running on Google Colab
# from google.colab import drive
# drive.mount('/content/drive')
# !unzip -o -q /content/drive/MyDrive/dataset.zip -d dataset
# DATADIR = 'dataset'
# OUTPUT = '/content/drive/MyDrive/output_ViT'
# print(DATADIR)

"""
# For running on Kaggle
# DATADIR = '/kaggle/input/coffee-leaf'
# OUTPUT = '/kaggle/working/'
# print(DATADIR)
# print(OUTPUT)
"""
DATADIR = 'dataset'
OUTPUT = ''
import os, sys, random, pathlib, time, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disable oneDNN custom ops which prints informational messages
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFile
import shutil
import cv2
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from sklearn.metrics import classification_report, confusion_matrix
from transformers import ViTForImageClassification, ViTConfig
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter # Tính tần suất lớp trong train
from tqdm.auto import tqdm # Import tqdm
import pickle
import datetime
import logging
from torch.amp import autocast
import tensorflow as tf
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

print('Đường dẫn dữ liệu:', DATADIR)
print('PyTorch:', torch.__version__)
if torch.cuda.is_available():
    t = torch.cuda.get_device_properties(0)
    print('GPU:', torch.cuda.get_device_name(0))
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Total VRAM: {t.total_memory / (1024**3):.2f} GB")
else:
    print(' Chạy trên CPU — sẽ chậm hơn.')

print(f"CUDA Available on PC: {torch.cuda.is_available()}")

# Performance-related defaults (safe opt-in)
USE_TORCH_COMPILE = True  # Set True to try torch.compile() (PyTorch 2.x). Test for stability first.
USE_CHANNELS_LAST = True   # Use channels_last memory format for faster conv/matmul on many GPUs.

# Enable cuDNN benchmark when input sizes are fixed (can improve throughput). Safe to enable.
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

"""## 2) Cấu hình & seed"""

print(f"Bắt đầu quá trình lúc: {datetime.datetime.now()}")
# Seed tái lập
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Cấu hình
DATASET_DIR = Path(DATADIR)
# Nếu thư mục dataset gốc không tồn tại nhưng processed_dataset đã có, dùng processed_dataset
if not DATASET_DIR.exists():
    alt = Path('processed_dataset')
    if alt.exists():
        DATASET_DIR = alt
        print(f"Thư mục {DATADIR} không tồn tại, dùng {alt} thay thế.")
    else:
        raise FileNotFoundError(f'Không tìm thấy thư mục {DATASET_DIR.resolve()}')
# Use HuggingFace ViT model from google (patch16 224)
MODEL_NAME = 'google/vit-base-patch16-224'
local_model_dir = Path(__file__).parent / "vit-base-patch16-224"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
BASE_LR = 5e-4
WEIGHT_DECAY = 0.02
WARMUP_EPOCHS = 2
LABEL_SMOOTHING = 0.05
MIXUP = True
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 0.0
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_PROCESSED = True
ImageFile.LOAD_TRUNCATED_IMAGES = True  # đọc được một số JPEG bị cắt ngang
print('Thiết bị:', DEVICE)
print('NUM_WORKERS:', NUM_WORKERS)
print('PIN_MEMORY:', PIN_MEMORY)

ALLOWED_EXTS = {'.jpg', '.jpeg', '.png'}

"""## 3) Liệt kê dữ liệu & tách train/val/test (stratified theo lớp) + lọc ảnh lỗi"""

# Nếu bạn muốn sử dụng lại các split đã có trong processed_dataset, bật USE_PROCESSED = True ở trên.
processed_dir = Path('processed_dataset')
if USE_PROCESSED and (processed_dir.exists() and (processed_dir / 'train').exists()):
    print(f"Tìm thấy thư mục processed_dataset, sẽ dùng split đã có và bỏ qua bước tách/lọc.")
    train_dir = processed_dir / 'train'
    val_dir = processed_dir / 'val'
    test_dir = processed_dir / 'test'
    CLASS_NAMES = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    NUM_CLASSES = len(CLASS_NAMES)
    print('Lớp (processed):', CLASS_NAMES)
    print('Số lớp:', NUM_CLASSES)
    # Hàm lấy danh sách ảnh và label từ thư mục
    def list_images_from_dir_existing(directory):
        paths, labels = [], []
        for idx, cls in enumerate(CLASS_NAMES):
            cls_dir = directory / cls
            files = [str(p) for p in cls_dir.rglob('*') if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
            paths.extend(files)
            labels.extend([idx] * len(files))
        return paths, labels
    train_paths, train_labels = list_images_from_dir_existing(train_dir)
    val_paths, val_labels = list_images_from_dir_existing(val_dir)
    test_paths, test_labels = list_images_from_dir_existing(test_dir)
    print(f'Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}')
else:
    CLASS_NAMES = sorted([p.name for p in DATASET_DIR.iterdir() if p.is_dir()])
    NUM_CLASSES = len(CLASS_NAMES)
    print('Lớp:', CLASS_NAMES)
    print('Số lớp:', NUM_CLASSES)

    output_data_dir = Path(OUTPUT + 'processed_dataset')
    train_dir = output_data_dir / 'train'
    val_dir = output_data_dir / 'val'
    test_dir = output_data_dir / 'test'

    def is_valid_image(path):
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return False
        try:
            with Image.open(p) as img:
                img.verify()
            return True
        except Exception:
            return False

    def list_images_by_class(src_dir):
        classes = sorted([d.name for d in Path(src_dir).iterdir() if d.is_dir()])
        per_class = {}
        for cls in tqdm(classes, desc="Đang xử lý lớp"):
            cls_dir = Path(src_dir) / cls
            files = [str(p) for p in cls_dir.rglob('*') if p.suffix.lower() in ALLOWED_EXTS]
            valid_files = [f for f in files if is_valid_image(f)]
            per_class[cls] = valid_files
            print(f"{cls}: {len(valid_files)} ảnh hợp lệ")
        return per_class
    def stratified_split(per_class, seed=SEED, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        rng = np.random.default_rng(seed)
        train, val, test = [], [], []
        for cls, files in per_class.items():
            idx = np.arange(len(files))
            rng.shuffle(idx)
            files = [files[i] for i in idx]
            n = len(files)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train += [(f, cls) for f in files[:n_train]]
            val += [(f, cls) for f in files[n_train:n_train+n_val]]
            test += [(f, cls) for f in files[n_train+n_val:]]
        return train, val, test
    def copy_files(pairs, out_dir):
        for path, cls in tqdm(pairs, desc=f"Copy to {out_dir.name}"):
            dst_dir = out_dir / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst_dir / Path(path).name)

    train_paths, val_paths, test_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    print(f'train: {train_paths} | val: {val_paths} | test: {test_paths}')

    seed=SEED
    src = Path(DATADIR)
    out = Path(OUTPUT + 'processed_dataset')
    out.mkdir(parents=True, exist_ok=True)

    print("Đang lọc ảnh hợp lệ...")
    per_class = list_images_by_class(src)

    print("\nĐang tách train/val/test...")
    train, val, test = stratified_split(per_class, seed)

    print(f"Tổng train: {len(train)}, val: {len(val)}, test: {len(test)}")

    print("\nĐang sao chép ảnh...")
    copy_files(train, out / 'train')
    copy_files(val, out / 'val')
    copy_files(test, out / 'test')

    print("\nHoàn tất! Dữ liệu đã lưu tại:", out)
    print(f"Tách và lọc ảnh hoàn tất lúc: {datetime.datetime.now()}")
"""## 4) Dataset & Transforms (augmentation) + fallback OpenCV"""
# Chuẩn hoá theo ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# RandAugment nếu có, fallback ColorJitter nếu không
try:
    rand_aug = [transforms.RandAugment(num_ops=2, magnitude=9)]
except Exception as e:
    print('RandAugment không khả dụng, dùng ColorJitter thay thế.')
    rand_aug = [transforms.ColorJitter(0.2,0.2,0.2,0.1)]

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    *rand_aug,
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

eval_tfms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def safe_open_image(path):
    # Thử PIL trước
    try:
        img = Image.open(path).convert('RGB')
        return img
    except Exception:
        pass
    # Fallback OpenCV
    img_cv = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise ValueError(f'Không thể đọc ảnh: {path}')
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_cv)

class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        # Placeholder image used when an image file cannot be opened
        try:
            from PIL import Image as _Image
            self._placeholder = _Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        except Exception:
            self._placeholder = None
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        y = self.labels[idx]
        try:
            img = safe_open_image(p)
        except Exception as e:
            # Nếu không thể mở ảnh, log và sử dụng ảnh placeholder để không làm crash DataLoader
            print(f"Warning: Không thể đọc ảnh {p}: {e}. Sử dụng ảnh placeholder.")
            if self._placeholder is not None:
                img = self._placeholder.copy()
            else:
                # Cuối cùng nếu không có placeholder thì raise để người dùng biết
                raise e
        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                # Nếu transform thất bại trên ảnh (ví dụ kích thước bất thường), dùng placeholder tensor
                print(f"Warning: Transform thất bại cho {p}: {e}. Dùng ảnh placeholder đã chuẩn hoá.")
                if self._placeholder is not None:
                    img = self.transform(self._placeholder.copy())
                else:
                    raise e
        return img, y

# Update: Load data from the processed_dataset directories
output_data_dir = Path(OUTPUT + 'processed_dataset')
train_dir = output_data_dir / 'train'
val_dir = output_data_dir / 'val'
test_dir = output_data_dir / 'test'

# Function to list images and labels from a directory structure
def list_images_from_dir(directory):
    paths = []
    labels = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = directory / cls_name
        files = [str(p) for p in cls_dir.rglob('*') if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
        paths.extend(files)
        labels.extend([cls_idx] * len(files))
    return paths, labels

train_paths, train_labels = list_images_from_dir(train_dir)
val_paths, val_labels = list_images_from_dir(val_dir)
test_paths, test_labels = list_images_from_dir(test_dir)

train_ds = ImageDataset(train_paths, train_labels, train_tfms)
val_ds = ImageDataset(val_paths, val_labels, eval_tfms)
test_ds = ImageDataset(test_paths, test_labels, eval_tfms)
print('Số mẫu train/val/test:', len(train_ds), len(val_ds), len(test_ds))

"""## 5) Xử lý mất cân bằng: WeightedRandomSampler (oversampling lớp nhỏ)"""

cnt = Counter(train_labels)
print('Phân bố train:', {CLASS_NAMES[k]: v for k,v in cnt.items()})

# Trọng số mẫu = 1 / tần suất lớp
class_freq = np.array([cnt.get(i, 0) for i in range(NUM_CLASSES)], dtype=np.float64)
class_freq[class_freq==0] = 1
class_weight = 1.0 / class_freq
sample_weights = np.array([class_weight[y] for y in train_labels], dtype=np.float64)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
print('Đã tạo WeightedRandomSampler (oversampling lớp ít).')

"""## 6) DataLoaders + MixUp"""

# MixUp setup
mixup_fn = None
if MIXUP:
    mixup_fn = Mixup(mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA,
                     prob=1.0, switch_prob=0.0, mode='batch',
                     label_smoothing=LABEL_SMOOTHING, num_classes=NUM_CLASSES)
    print(' MixUp đang bật')
else:
    print(' MixUp tắt')

def collate_fn(batch):
    imgs, labels = list(zip(*batch))
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    if mixup_fn is not None:
        imgs, labels = mixup_fn(imgs, labels)
    return imgs, labels

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
len(train_loader), len(val_loader), len(test_loader)

"""## 7) Khởi tạo model"""

use_local = local_model_dir.exists()
try:
    if use_local:
        print(f"Tìm thấy thư mục mô hình local: {local_model_dir}. Sẽ load từ local với num_labels={NUM_CLASSES}\n")
        # Gán mapping nhãn theo tập lớp hiện có
        id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
        label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
        model = ViTForImageClassification.from_pretrained(
            str(local_model_dir),
            num_labels=NUM_CLASSES,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        if getattr(model.config, "num_labels", None) != NUM_CLASSES or (hasattr(model, "classifier") and getattr(model.classifier, "out_features", None) != NUM_CLASSES):
            hidden = model.config.hidden_size
            model.classifier = nn.Linear(hidden, NUM_CLASSES)
            model.config.num_labels = NUM_CLASSES
            model.config.id2label = id2label
            model.config.label2id = label2id
    else:
        print(f"Không tìm thấy model local, sẽ tải '{MODEL_NAME}' từ HuggingFace (cần mạng) với num_labels={NUM_CLASSES}")
        model = ViTForImageClassification.from_pretrained(MODEL_NAME)

    # Nếu kích thước bộ phân loại khác, thay head phân lớp bằng kích thước đúng
    try:
        model_config = model.config
        if getattr(model_config, 'num_labels', None) != NUM_CLASSES:
            print(f"Đang điều chỉnh head từ {model_config.num_labels} sang {NUM_CLASSES} nhãn")
            hidden = model_config.hidden_size
            model.classifier = torch.nn.Linear(hidden, NUM_CLASSES)
            model.config.num_labels = NUM_CLASSES
            model.config.id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
            model.config.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
    except Exception:
        pass

except Exception as e:
    print('Lỗi khởi tạo mô hình HF ViT hoặc tải trọng số:', e)
    # Fallback: tạo mô hình mới với cấu hình (cố gắng lấy config từ local nếu có)
    try:
        if use_local:
            cfg = ViTConfig.from_pretrained(str(local_model_dir), local_files_only=True)
        else:
            cfg = ViTConfig.from_pretrained(MODEL_NAME)
    except Exception:
        cfg = ViTConfig()
    cfg.num_labels = NUM_CLASSES
    model = ViTForImageClassification(cfg)
    model.config.id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    model.config.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

model.to(DEVICE)
print('Số tham số huấn luyện:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# Loss: nếu MixUp -> SoftTargetCrossEntropy; nếu không -> CrossEntropy có label smoothing
if MIXUP:
    criterion = SoftTargetCrossEntropy()
else:
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS))
if WARMUP_EPOCHS > 0:
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_EPOCHS])
else:
    scheduler = main_scheduler

scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

"""## 8) Vòng lặp huấn luyện"""

def accuracy_from_logits(logits, targets):
    if logits.ndim == 2 and targets.ndim == 2:
        preds = logits.argmax(dim=1)
        y = targets.argmax(dim=1)
    else:
        preds = logits.argmax(dim=1)
        y = targets
    return (preds == y).float().mean().item()

def train_one_epoch(epoch):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    n_batches = 0
    print(f"\nBắt đầu epoch {epoch}/{EPOCHS} lúc: {datetime.datetime.now()}")

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]", leave=False)
    for imgs, labels in progress_bar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = model(imgs)
            # HuggingFace models return ImageClassifierOutput; logits tensor is in .logits
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc = accuracy_from_logits(logits.detach(), labels.detach())
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

        progress_bar.set_postfix(loss=loss.item(), acc=acc)

    return total_loss / max(1,n_batches), total_acc / max(1,n_batches)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    eval_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    progress_bar = tqdm(loader, desc="[EVAL]", leave=False)
    for imgs, labels in progress_bar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE)
        with autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = model(imgs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = eval_criterion(logits, labels)
        acc = accuracy_from_logits(logits, labels)
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1
        progress_bar.set_postfix(loss=loss.item(), acc=acc)

    return total_loss / max(1,n_batches), total_acc / max(1,n_batches)

best_val_acc = -1.0
patience, patience_count = EPOCHS / 2, 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
ckpt_dir = Path(OUTPUT + 'checkpoints')
ckpt_dir.mkdir(parents=True, exist_ok=True)
best_ckpt_path = ckpt_dir / 'best_model.pth'

for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(epoch)
    val_loss, val_acc = evaluate(val_loader)
    scheduler.step()
    history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
    history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
    dt = time.time()-t0
    print(f'Epoch {epoch}/{EPOCHS} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.1f}s')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_count = 0
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, best_ckpt_path)
        print('********************************* Lưu best model:', best_ckpt_path)
    else:
        patience_count += 1
        if patience_count >= patience:
            print(f'Dừng huấn luyện sau {patience} epoch không cải thiện.')
            break

# Lưu lịch sử
with open(OUTPUT + 'checkpoints/training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
print(' Huấn luyện xong. Lịch sử lưu vào training_history.pkl')

"""## 9) Biểu đồ Loss/Accuracy"""

with open(OUTPUT + 'training_history.pkl', 'rb') as f:
    history = pickle.load(f)
epochs_ran = range(1, len(history['train_loss'])+1)

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(1,2,1)
ax1.plot(epochs_ran, history['train_loss'], label='Train Loss')
ax1.plot(epochs_ran, history['val_loss'],   label='Val Loss')
ax1.set_title('Biểu đồ Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()

ax2 = fig.add_subplot(1,2,2)
ax2.plot(epochs_ran, history['train_acc'], label='Train Acc')
ax2.plot(epochs_ran, history['val_acc'],   label='Val Acc')
ax2.set_title('Biểu đồ Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend()

fig.tight_layout()

out_plot = Path(OUTPUT) / 'checkpoints/loss_accuracy_plot.png'
fig.savefig(out_plot, bbox_inches='tight', dpi=200)
print(f"Biểu đồ Loss và Accuracy đã được lưu tại {out_plot}")

plt.show()
plt.close(fig)

"""## 10) Đánh giá trên Test + Báo cáo chi tiết"""

# Tải best model
best_ckpt_path = Path(OUTPUT + 'models/best_model.pth')
assert best_ckpt_path.exists(), 'Chưa có best checkpoint'
ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()

# Evaluate
@torch.no_grad()
def predict_loader(loader):
    y_true, y_pred = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        y_pred += preds
        y_true += labels.numpy().tolist()
    return y_true, y_pred

y_true, y_pred = predict_loader(test_loader)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print('Báo cáo phân loại (Test):')
print(report)

# Save classification report
report_path = Path(OUTPUT + 'checkpoints/classification_report.txt')
with open(report_path, 'w') as f:
    f.write(report)
print(f"Báo cáo phân loại đã được lưu tại {report_path}")

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title('Ma trận nhầm lẫn (Test)'); ax.set_xlabel('Dự đoán'); ax.set_ylabel('Thực tế')
fig.tight_layout()

cm_path = Path(OUTPUT) / 'confusion_matrix.png'
fig.savefig(cm_path, bbox_inches='tight', dpi=200)
print(f"Ma trận nhầm lẫn đã được lưu tại {cm_path}")

plt.show()
plt.close(fig)

"""## 11) Lưu mô hình cuối cùng"""

save_dir = Path(OUTPUT + 'saved_model')
save_dir.mkdir(parents=True, exist_ok=True)
final_path = save_dir / 'vit_coffee_leaf_disease.pth'
torch.save({'model': model.state_dict(), 'classes': CLASS_NAMES}, final_path)
print(' Đã lưu mô hình vào', final_path)

print(f"Kết thúc quá trình lúc: {datetime.datetime.now()}")

"""---
### Gợi ý tối ưu thêm
- Thử `MIXUP_ALPHA=0.3~0.4` nếu còn overfitting.
- Tăng `IMG_SIZE` lên 256/288 nếu GPU cho phép (đổi model sang `vit_base_patch16_384` nếu muốn 384).
- Số epoch có thể 30–50 để khai thác ViT tốt hơn (với Cosine LR), nhưng theo yêu cầu đang đặt là 20.
- Nếu bạn muốn **fine-tune theo 2 giai đoạn** (warmup head rồi mới unfreeze), có thể thay đổi: freeze tất cả, train head 3–5 epochs, sau đó unfreeze toàn bộ và giảm LR.

"""