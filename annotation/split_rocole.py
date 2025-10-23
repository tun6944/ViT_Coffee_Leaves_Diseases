
# split_rocole.py
# Tách ảnh RoCoLe (Photos + Annotation) thành các thư mục theo bệnh: healthy / rust / red_spider_mite
# Sử dụng nhãn từ file Excel (RoCoLe-classes.xlsx) hoặc CSV export (Labelbox CSV).

import argparse
import pandas as pd
from pathlib import Path
import shutil
import sys
photos_dir = 'Photos'
output_dir = 'rocole_split'
MAP_MULTICLASS = {
    'healthy': 'healthy',
    'rust_level_1': 'rust',
    'rust_level_2': 'rust',
    'rust_level_3': 'rust',
    'rust_level_4': 'rust',
    'red_spider_mite': 'red_spider_mite'
}


def load_labels_from_xlsx(xlsx_path: Path):
    # Expect columns: File, Binary.Label, Multiclass.Label
    df = pd.read_excel(xlsx_path)
    if not set(['File', 'Binary.Label', 'Multiclass.Label']).issubset(df.columns):
        raise ValueError('File Excel không có các cột bắt buộc: File, Binary.Label, Multiclass.Label')
    return df[['File', 'Binary.Label', 'Multiclass.Label']]


def load_labels_from_csv(csv_path: Path):
    # Labelbox export: có trường 'Labeled Data' (URL) và chuỗi JSON trong cột 'Label' chứa 'classification'
    # Một số export khác có cột 'External ID' hoặc tên file trong URL.
    df = pd.read_csv(csv_path)
    if 'Label' not in df.columns:
        raise ValueError('CSV không có cột Label (Labelbox export)')
    # Trích xuất filename từ URL hoặc External ID
    if 'External ID' in df.columns:
        filenames = df['External ID'].astype(str)
    elif 'Labeled Data' in df.columns:
        filenames = df['Labeled Data'].apply(lambda x: Path(str(x)).name)
    else:
        raise ValueError('CSV cần có External ID hoặc Labeled Data để suy ra tên file')

    # Lấy classification
    import json
    def parse_class(row):
        try:
            obj = json.loads(row)
            cls = obj.get('classification')
            return cls
        except Exception:
            return None
    classes = df['Label'].apply(parse_class)

    out = pd.DataFrame({'File': filenames, 'Multiclass.Label': classes})
    # Nếu thiếu class -> bỏ
    out = out.dropna(subset=['Multiclass.Label'])
    # Chuẩn hoá
    out['Multiclass.Label'] = out['Multiclass.Label'].str.lower()
    return out


def main():
    parser = argparse.ArgumentParser(description='Tách ảnh RoCoLe theo nhãn bệnh')
    parser.add_argument('--photos_dir', type=str, required=True, help='Photos')
    parser.add_argument('--xlsx', type=str, default=None, help='Annotations\RoCoLe-classes.xlsx')
    parser.add_argument('--label_csv', type=str, default=None, help='Annotations\RoCoLE-csv.csv')
    parser.add_argument('--output_dir', type=str, default='rocole_split', help='output_dir')
    args = parser.parse_args()

    photos_dir = Path(args.photos_dir)
    assert photos_dir.exists(), f'Không thấy thư mục {photos_dir}'

    if args.xlsx:
        df = load_labels_from_xlsx(Path(args.xlsx))
    elif args.label_csv:
        df = load_labels_from_csv(Path(args.label_csv))
        df['Binary.Label'] = None
    else:
        print('Cần truyền --xlsx hoặc --label_csv')
        sys.exit(1)

    # Ưu tiên Multiclass.Label nếu có; nếu không thì Binary.Label (unhealthy -> diseased)
    def map_row(row):
        m = str(row.get('Multiclass.Label') or '').strip().lower()
        b = str(row.get('Binary.Label') or '').strip().lower()
        if m in MAP_MULTICLASS:
            return MAP_MULTICLASS[m]
        if b == 'healthy':
            return 'healthy'
        if b == 'unhealthy':
            return 'diseased'
        return 'unknown'

    df['class'] = df.apply(map_row, axis=1)

    out_dir = Path(args.output_dir)
    for cls in sorted(set(df['class'])):
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    copied, missing = 0, []
    for _, r in df.iterrows():
        fname = r['File']
        cls = r['class']
        src = photos_dir / fname
        if not src.exists():
            # thử tìm case-insensitive
            low = fname.lower()
            cand2 = [p for p in photos_dir.iterdir() if p.is_file() and p.name.lower() == low]
            if not cand2:
                missing.append(fname)
                continue
            src = cand2[0]
        dst = out_dir / cls / src.name
        shutil.copy2(src, dst)
        copied += 1

    print(f'Đã copy {copied} ảnh vào {out_dir}')
    if missing:
        print(f'⚠️ Không tìm thấy {len(missing)} ảnh trong {photos_dir}. Ví dụ: {missing[:10]}')

if __name__ == '__main__':
    main()
