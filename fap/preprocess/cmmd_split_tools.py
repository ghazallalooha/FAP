import os, shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_labels_csv(full_csv, out_dir, test_size=0.2, val_fraction_of_train=0.125, seed=42):
    """
    Split a single CMMD labels CSV (image_path,label) into train/val/test CSVs in out_dir.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(full_csv)
    X = df['image_path']; y = df['label']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=val_fraction_of_train, stratify=y_tr, random_state=seed)
    for split, (X_, y_) in {'train':(X_tr,y_tr),'val':(X_va,y_va),'test':(X_te,y_te)}.items():
        d = out / split; d.mkdir(exist_ok=True)
        pd.DataFrame({'image_path': X_, 'label': y_}).to_csv(d / f"{split}_labels.csv", index=False)

def copy_split_images_unique(base_dir):
    """
    Copy images referenced by each split CSV into split-specific folders with unique flattened names.
    """
    base = Path(base_dir)
    for split in ['train','val','test']:
        csv_path = base / split / f"{split}_labels.csv"
        img_out = base / f"{split}_images"; img_out.mkdir(exist_ok=True)
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            src = Path(r['image_path'])
            dst = img_out / ("__".join(src.parts[-3:]))  # flatten last 3 levels
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"[WARN] copy {src} -> {dst}: {e}")
