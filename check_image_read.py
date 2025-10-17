# check_image.py
import os
from pathlib import Path
from PIL import Image, ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True

def safe_open_image_try(path):
    try:
        with Image.open(path) as im:
            im.verify()
        # reopen to convert (verify() can leave file in unusable state)
        with Image.open(path) as im:
            im = im.convert('RGB')
        return True, None
    except Exception as e_pil:
        # try OpenCV
        try:
            img_cv = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img_cv is None:
                raise ValueError("cv2 returned None")
            return True, None
        except Exception as e_cv:
            return False, (e_pil, e_cv)

if __name__ == "__main__":
    bad_list = []
    root = Path('processed_dataset')
    # 1) test single problematic file
    p = Path(r'processed_dataset\\train\\phoma\\6 (2955).jpg')
    ok, err = safe_open_image_try(p)
    print("Single file check:", p, "OK?", ok, "Err:", err)

    # 2) quick scan (stop after finding N problems for speed)
    max_errors = 200
    found = 0
    for fp in root.rglob('*'):
        if not fp.is_file(): continue
        ok, err = safe_open_image_try(fp)
        if not ok:
            print("Bad:", fp, "Err:", err)
            bad_list.append(fp)
            found += 1
            if found >= max_errors:
                break

    print(f"Found {len(bad_list)} bad files (showing up to {max_errors}).")
    # optional: move them to a safe folder to avoid training crash
    if bad_list:
        out = root / 'bad_images'
        out.mkdir(exist_ok=True)
        for b in bad_list:
            dest = out / b.name
            try:
                b.rename(dest)  # move
            except Exception:
                try:
                    import shutil
                    shutil.copy2(b, dest)
                    b.unlink()
                except Exception as e:
                    print("Failed to move:", b, e)
        print("Moved bad files to", out)