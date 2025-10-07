import os, re, pydicom, pandas as pd
from PIL import Image
from pathlib import Path

def extract_filename_from_path(path):
    if pd.isna(path): return None
    m = re.search(r'(Calc|Mass)-(Test|Training)_(P_\d+_\w+_\w+)', path)
    if not m: return None
    return f"P_{m.group(2)}_{m.group(3)}"

def create_label_mapping(excel_csv_path, output_dir):
    df = pd.read_csv(excel_csv_path) if excel_csv_path.lower().endswith(".csv") else pd.read_excel(excel_csv_path)
    col = 'image file path' if 'image file path' in df.columns else 'cropped image file path'
    if col not in df.columns:
        print(f"[WARN] no image path column in {excel_csv_path}"); return
    df['filename'] = df[col].apply(extract_filename_from_path)
    df = df.dropna(subset=['filename'])
    df['label'] = df['pathology'].map({'BENIGN':0,'MALIGNANT':1,'BENIGN_WITHOUT_CALLBACK':0})
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df[['filename','label']].to_csv(os.path.join(output_dir, 'label_mapping.csv'), index=False)

def convert_dcm_to_png(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.dcm'):
                dcm_path = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(dcm_path)
                    px = ds.pixel_array
                    if px.ndim > 2: px = px[0]
                    img = Image.fromarray(px)
                    png_path = os.path.join(output_dir, os.path.splitext(f)[0] + ".png")
                    img.save(png_path)
                except Exception as e:
                    print(f"[WARN] {dcm_path}: {e}")
