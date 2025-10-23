#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_rust_miner_xml.py
Tách ảnh theo nhãn từ bộ dữ liệu VOC với XML annotation.
- Mặc định quét hai thư mục: ./rust_xml_image và ./miner_img_xml (có thể chỉ có 1 trong 2)
- Phân lớp: 'rust', 'miner', 'mixed', 'unknown'
- Output: ./rust_miner_split
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
import sys

# Các phần mở rộng ảnh hợp lệ
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Bảng synonym để chuẩn hoá nhãn XML về lớp chuẩn
# (bổ sung thêm nếu dataset có biến thể khác)
SYN_MAP = {
    # rust
    'ferrugem': 'rust',         # PT: rỉ sắt (name1.xml) 
    'coffee_rust': 'rust',
    'roya': 'rust',             # ES: rỉ sắt
    'royadelcafé': 'rust',
    'roya_del_café': 'rust',
    'rust': 'rust',

    # leaf miner
    'bicho_mineiro': 'miner',   # PT: bọ đục lá (bicho_mineiro0.xml)
    'leaf_miner': 'miner',
    'leafminer': 'miner',
    'minador': 'miner',         # ES/PT: “minador”
    'minadora': 'miner',
    'lagarta-minadora': 'miner',
    'broca_minadora': 'miner',
    'miner': 'miner',
}

def norm_name(name: str) -> str:
    """Chuẩn hoá tên object từ XML trước khi map."""
    n = (name or '').strip().lower()
    # Thống nhất ký tự: thay khoảng trắng/thanh ngang thành underscore
    for ch in [' ', '-', '\t', '\n']:
        n = n.replace(ch, '_')
    # Bỏ các underscore thừa liền nhau
    while '__' in n:
        n = n.replace('__', '_')
    return SYN_MAP.get(n, n)

def parse_classes(xml_path: Path) -> set:
    """Trả về tập tên lớp đã chuẩn hoá xuất hiện trong XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    names = set()
    for obj in root.findall('.//object'):
        name = obj.findtext('name')
        if not name:
            continue
        names.add(norm_name(name))
    return names

def extract_filename_from_xml(xml_path: Path) -> str | None:
    """Lấy <filename> từ XML nếu có."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        fn = root.findtext('.//filename')
        if fn:
            return fn.strip()
    except Exception:
        pass
    return None

def find_image_for_xml(xml_path: Path, search_roots: list[Path]) -> Path | None:
    """
    Tìm ảnh ứng với XML theo các chiến lược:
    1) Cùng thư mục (stem giống),
    2) Dùng <filename> trong XML,
    3) Quét các root được cung cấp.
    """
    stem = xml_path.stem
    # 1) Stem + ext trong cùng thư mục
    for ext in VALID_EXTS:
        cand = xml_path.with_name(stem + ext)
        if cand.exists():
            return cand
    # 2) Dùng <filename>
    xml_filename = extract_filename_from_xml(xml_path)
    # 2a) nếu có filename, thử ngay cùng thư mục
    if xml_filename:
        cand = xml_path.parent / xml_filename
        if cand.exists():
            return cand
    # 3) Quét trong các root
    candidates = []
    # 3a) nếu có filename, ưu tiên tìm theo filename (case-insensitive)
    if xml_filename:
        for root in search_roots:
            # Trực tiếp theo tên
            direct = root / xml_filename
            if direct.exists():
                return direct
            # Case-insensitive trong root
            files_lower = {p.name.lower(): p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in VALID_EXTS}
            hit = files_lower.get(xml_filename.lower())
            if hit:
                return hit
    # 3b) nếu không có filename hoặc chưa thấy, tìm theo stem + ext trong từng root
    for root in search_roots:
        for ext in VALID_EXTS:
            cand = root / (stem + ext)
            if cand.exists():
                return cand
        # Tìm theo stem.* trong cây
        for p in root.rglob(stem + '.*'):
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                return p
    return None

def decide_class(names: set) -> str:
    """
    Quy tắc phân lớp:
    - chỉ 'rust'  -> 'rust'
    - chỉ 'miner' -> 'miner'
    - có cả hai   -> 'mixed'
    - khác/không rõ -> 'unknown'
    """
    has_rust = ('rust' in names)
    has_miner = ('miner' in names)
    if not names:
        return 'unknown'
    if has_rust and not has_miner:
        return 'rust'
    if has_miner and not has_rust:
        return 'miner'
    if has_rust and has_miner:
        return 'mixed'
    return 'unknown'

def main():
    parser = argparse.ArgumentParser(description='Tách ảnh theo nhãn từ XML (VOC) với synonym PT/ES/EN')
    parser.add_argument('--roots', type=str, nargs='+', default=None,
                        help='Một hoặc nhiều thư mục gốc để quét (ví dụ: rust_xml_image miner_img_xml)')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Chỉ định 1 thư mục gốc (tùy chọn)')
    parser.add_argument('--output_dir', type=str, default='rust_miner_split',
                        help='Thư mục đầu ra')
    parser.add_argument('--copy_xml', action='store_true',
                        help='Nếu bật, copy kèm file XML vào cùng thư mục lớp')
    args = parser.parse_args()

    # Xác định roots
    if args.roots:
        roots = [Path(p) for p in args.roots]
    elif args.root_dir:
        roots = [Path(args.root_dir)]
    else:
        roots = [Path('rust_xml_image'), Path('miner_img_xml')]  # mặc định theo yêu cầu bạn

    roots = [p for p in roots if p.exists()]
    if not roots:
        print('❌ Không tìm thấy thư mục nào trong: ./rust_xml_image, ./miner_img_xml')
        print('→ Truyền --roots /path1 /path2 hoặc --root_dir /path')
        sys.exit(2)

    # Thư mục output
    out = Path(args.output_dir)
    for cls in ['rust', 'miner', 'mixed', 'unknown']:
        (out / cls).mkdir(parents=True, exist_ok=True)

    # Thu thập mọi XML
    xmls = []
    for r in roots:
        xmls.extend(r.rglob('*.xml'))

    if not xmls:
        print('⚠️ Không tìm thấy file .xml trong', ', '.join(str(x) for x in roots))
        return

    copied = 0
    missing_img = []
    for xml_path in xmls:
        # Lấy tập tên lớp đã chuẩn hoá
        names = parse_classes(xml_path)
        cls = decide_class(names)

        # Tìm ảnh đi kèm
        img = find_image_for_xml(xml_path, search_roots=roots)
        if not img:
            missing_img.append(xml_path.name)
            continue

        # Copy ảnh vào thư mục lớp
        dst_img = out / cls / img.name
        shutil.copy2(img, dst_img)

        # (tuỳ chọn) Copy XML đi kèm
        if args.copy_xml:
            dst_xml = out / cls / xml_path.name
            shutil.copy2(xml_path, dst_xml)

        copied += 1

    print(f'✅ Đã copy {copied} ảnh vào {out}')
    if missing_img:
        print(f'⚠️ {len(missing_img)} XML không tìm thấy ảnh kèm theo. Ví dụ: {missing_img[:10]}')

if __name__ == '__main__':
    main()