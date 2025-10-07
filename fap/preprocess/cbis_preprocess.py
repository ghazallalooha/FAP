import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from fap.preprocess.cbis_tools import create_label_mapping, convert_dcm_to_png

def build_cbis_from_dicom(dicom_root: str, labels_excel_csv: str, out_root: str,
                          png_dirname: str = "images_png", seed: int = 42):
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    png_dir = out_root / png_dirname; png_dir.mkdir(parents=True, exist_ok=True)
    convert_dcm_to_png(dicom_root, str(png_dir))
    create_label_mapping(labels_excel_csv, str(out_root))
    lbl = pd.read_csv(out_root / "label_mapping.csv")
    avail = {p.name: str(p) for p in png_dir.glob("*.png")}
    rows = []
    for _, r in lbl.iterrows():
        fname = r["filename"]
        lab = int(r["label"])
        png = fname + ".png"
        if png in avail:
            rows.append({"image_path": avail[png], "label": lab})
    full = pd.DataFrame(rows)
    if full.empty:
        raise RuntimeError("No CBIS-DDSM PNGs matched label mapping; check filename conventions.")
    X_tr, X_te, y_tr, y_te = train_test_split(full["image_path"], full["label"], test_size=0.2, stratify=full["label"], random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.125, stratify=y_tr, random_state=seed)
    for split, (X, y) in {"train": (X_tr, y_tr), "val": (X_va, y_va), "test": (X_te, y_te)}.items():
        d = out_root / split; d.mkdir(exist_ok=True, parents=True)
        pd.DataFrame({"image_path": X, "label": y}).to_csv(d / f"{split}_labels.csv", index=False)
    print(f"[OK] CBIS prepared at {out_root}")
