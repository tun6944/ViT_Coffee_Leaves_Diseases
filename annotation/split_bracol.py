# split_bracol.py
# Mặc định: ảnh ./leaf/images, CSV ./leaf/dataset.csv, output ./bracol_split

import argparse
import pandas as pd
from pathlib import Path
import shutil
import re
import sys

MAP_STRESS = {
    0: 'healthy',
    1: 'miner',
    2: 'rust',
    3: 'phoma',
    4: 'cercospora',
    5: 'mixed'
}

def infer_filename(images_dir: Path, id_val: int):
    # Tìm file có chứa id_val trong tên (biên số)
    if id_val is None:
        return None
    pattern = re.compile(r'(?<!\d)'+str(id_val)+r'(?!\d)')
    for p in images_dir.rglob('*'):
        if p.is_file() and pattern.search(p.stem):
            return p
    # fallback: tên đúng bằng id
    for ext in ['.jpg', '.jpeg', '.png']:
        p = images_dir / f"{id_val}{ext}"
        if p.exists():
            return p
    return None

def main():
    parser = argparse.ArgumentParser(description='Tách ảnh BRACOL theo nhãn bệnh')
    parser.add_argument('--images_dir', type=str, default=None, help='Thư mục ảnh (mặc định: ./leaf/images)')
    parser.add_argument('--csv_path', type=str, default=None, help='Đường dẫn CSV (mặc định: ./leaf/dataset.csv)')
    parser.add_argument('--output_dir', type=str, default='bracol_split', help='Thư mục đầu ra')
    args = parser.parse_args()

    images_dir = Path(args.images_dir) if args.images_dir else Path('leaf/images')
    csv_path = Path(args.csv_path) if args.csv_path else Path('leaf/dataset.csv')
    out_dir = Path(args.output_dir)

    if not images_dir.exists():
        print('❌ Không thấy thư mục ảnh:', images_dir.resolve()); sys.exit(2)
    if not csv_path.exists():
        print('❌ Không thấy file CSV:', csv_path.resolve()); sys.exit(2)

    df = pd.read_csv(csv_path)
    if 'predominant_stress' not in df.columns:
        raise ValueError('CSV cần có cột predominant_stress')

    out_dir.mkdir(parents=True, exist_ok=True)
    created = set()
    def ensure_dir(c):
        if c not in created:
            (out_dir / c).mkdir(parents=True, exist_ok=True)
            created.add(c)

    for c in set(MAP_STRESS.values()):
        ensure_dir(c)

    # Nếu có cột filename thì ưu tiên dùng
    fname_col = next((c for c in ['filename', 'file', 'img', 'image', 'image_name'] if c in df.columns), None)

    copied, missing = 0, []
    for _, r in df.iterrows():
        # Chọn lớp
        try:
            cls = MAP_STRESS.get(int(r['predominant_stress']), 'unknown')
        except Exception:
            cls = 'unknown'

        if cls == 'mixed':
            mix = []
            for k in ['miner', 'rust', 'phoma', 'cercospora']:
                if k in df.columns:
                    try:
                        if int(r.get(k, 0)) == 1:
                            mix.append(k)
                    except Exception:
                        pass
            if mix:
                cls = 'mixed_' + '_'.join(sorted(mix))
                ensure_dir(cls)

        # Tìm ảnh nguồn
        src = None
        if fname_col:
            cand = images_dir / str(r[fname_col])
            if cand.exists():
                src = cand
        if src is None:
            id_val = None
            if 'id' in df.columns:
                try:
                    id_val = int(r['id'])
                except Exception:
                    id_val = None
            src = infer_filename(images_dir, id_val)

        if not src or not src.exists():
            missing.append(r.get('id', 'unknown'))
            continue

        dst = out_dir / cls / src.name
        shutil.copy2(src, dst)
        copied += 1

    print('✅ Đã copy', copied, 'ảnh vào', out_dir)
    if missing:
        print('⚠️ Không khớp filename cho', len(missing), 'dòng. Ví dụ:', missing[:10])

if __name__ == '__main__':
    main()
