import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

ALLOWED_EXTS = {'.jpg', '.jpeg', '.png'}

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
        print(f"\n{cls}: {len(valid_files)} ảnh hợp lệ")
    return per_class


def stratified_split(per_class, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
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

def main():
    seed=42
    src = Path('dataset')
    out = Path('processed_dataset')
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

if __name__ == "__main__":
    main()