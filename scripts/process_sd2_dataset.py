import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import json
import shutil

def extract_valid_crop(mask_arr, target_size=224, condition="fake"):
    h, w = mask_arr.shape
    if target_size > h or target_size > w: return None
    if condition == "fake":
        coords = np.argwhere(mask_arr > 0)
        if len(coords) == 0: return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
        top = max(0, min(cy - target_size // 2, h - target_size))
        left = max(0, min(cx - target_size // 2, w - target_size))
        return (left, top, left + target_size, top + target_size)
    elif condition == "real":
        for _ in range(50):
            top = random.randint(0, h - target_size)
            left = random.randint(0, w - target_size)
            if np.max(mask_arr[top:top+target_size, left:left+target_size]) == 0:
                return (left, top, left + target_size, top + target_size)
        return None

def process_split(split, mapping):
    random.seed(42)
    img_dir, mask_dir = mapping[split]
    
    out_dir = f"sd2-classification/{split}"
    fake_out = os.path.join(out_dir, "FAKE")
    real_out = os.path.join(out_dir, "REAL")
    mask_out = os.path.join(out_dir, "MASKS")
    os.makedirs(fake_out, exist_ok=True)
    os.makedirs(real_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    images = glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True) + \
             glob.glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True)
    
    fake_pool, real_pool = [], []
    
    for img_path in tqdm(images, desc=f"Scanning {split}"):
        filename = os.path.basename(img_path)
        sub_dir = os.path.basename(os.path.dirname(img_path))
        mask_base = filename.split("_sd2")[0]
        mask_base_512 = mask_base.replace(".png", "_512.png")
        mask_path = os.path.join(mask_dir, sub_dir, mask_base_512)
        if not os.path.exists(mask_path): continue
            
        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            mask_arr = np.array(mask)
        except Exception:
            continue
            
        fake_box = extract_valid_crop(mask_arr, condition="fake")
        if fake_box: fake_pool.append((img_path, mask_path, fake_box, sub_dir, filename, "fake"))
            
        real_box = extract_valid_crop(mask_arr, condition="real")
        if real_box: real_pool.append((img_path, None, real_box, sub_dir, filename, "real"))

    # Balance
    min_count = min(len(fake_pool), len(real_pool))
    random.shuffle(fake_pool)
    random.shuffle(real_pool)
    fake_pool = fake_pool[:min_count]
    real_pool = real_pool[:min_count]
    
    print(f"[{split}] Saving {min_count} REAL and {min_count} FAKE images...")
    
    metadata = {}
    for item in fake_pool + real_pool:
        i_path, m_path, box, s_dir, fname, cond = item
        out_name = f"{s_dir}_{fname}_{cond}.png"
        out_path = os.path.join(fake_out if cond == 'fake' else real_out, out_name)
        
        img = Image.open(i_path).convert("RGB")
        img.crop(box).save(out_path)
        box_list = [int(x) for x in box]
        
        meta_entry = {"label": "FAKE" if cond == "fake" else "REAL", "crop_bbox": box_list, "source": i_path}
        
        if cond == "fake" and m_path:
            mask_img = Image.open(m_path).convert("L")
            mask_crop = mask_img.crop(box)
            mask_out_name = f"{s_dir}_{fname}_mask.png"
            mask_crop.save(os.path.join(mask_out, mask_out_name))
            
            # Find the bounding box of the manipulation relative to this crop itself
            mask_arr = np.array(mask_crop)
            coords = np.argwhere(mask_arr > 0)
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                meta_entry["manipulation_bbox_in_crop"] = [int(x_min), int(y_min), int(x_max), int(y_max)]
            else:
                meta_entry["manipulation_bbox_in_crop"] = []
                
            meta_entry["mask_file"] = mask_out_name
            
        metadata[out_name] = meta_entry

    return min_count, metadata

def main():
    mappings = {
        "train": ("sd2-fr-training", "sd2-fr-training-masks"),
        "val": ("sd2-fr-validation", "sd2-fr-validation-masks"),
        "test": ("sd2-fr-testing", "sd2-fr-testing-masks")
    }
    
    if os.path.exists("sd2-classification"):
        shutil.rmtree("sd2-classification")
    
    all_metadata = {}
    readme_lines = ["# SD2 Classification Dataset\n", "| Split | FAKE count | REAL count | Total |\n|---|---|---|---|"]
    
    for split in mappings:
        count, meta = process_split(split, mappings)
        all_metadata[split] = meta
        readme_lines.append(f"| {split} | {count} | {count} | {count*2} |")
        
    with open("sd2-classification/metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=4)
        
    with open("sd2-classification/README.md", "w") as f:
        f.write("\n".join(readme_lines) + "\n")
        
    print("Done generating dataset. Deleting old raw source directories...")
    # Delete original large folders that we extracted from
    for img_d, msk_d in mappings.values():
        if os.path.exists(img_d): shutil.rmtree(img_d)
        if os.path.exists(msk_d): shutil.rmtree(msk_d)
    
    if os.path.exists("sd2-patches"): shutil.rmtree("sd2-patches")
    if os.path.exists("create_patch_dataset.py"): os.remove("create_patch_dataset.py")
    if os.path.exists("evaluate_sd2_patches.py"): os.remove("evaluate_sd2_patches.py")
    if os.path.exists("sd2_patches_evaluation.json"): os.remove("sd2_patches_evaluation.json")

if __name__ == "__main__":
    main()
