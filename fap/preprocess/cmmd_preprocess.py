import os, cv2, pydicom
import pandas as pd
from sklearn.model_selection import train_test_split

def convert_cmmd_dataset(dicom_root: str, excel_path: str, output_dir: str):
    """
    Convert CMMD DICOMs -> JPG; make train/val/test CSVs with columns [image_path,label].
    """
    df = pd.read_excel(excel_path)
    label_map = {row['ID1']: 0 if str(row['classification']).strip().lower()=='benign' else 1
                 for _, row in df.iterrows()}

    os.makedirs(output_dir, exist_ok=True)
    image_paths, labels = [], []

    for patient_dir in os.listdir(dicom_root):
        if not patient_dir.startswith('D'): continue
        patient_path = os.path.join(dicom_root, patient_dir)
        for root, _, files in os.walk(patient_path):
            for f in files:
                if not f.lower().endswith('.dcm'): continue
                dcm_path = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(dcm_path)
                    px = ds.pixel_array
                    if px.ndim > 2: px = px[0]
                    img = cv2.normalize(px, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                    out_dir = os.path.join(output_dir, patient_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"{os.path.splitext(f)[0]}.jpg")
                    cv2.imwrite(out_path, img)
                    if patient_dir in label_map:
                        image_paths.append(out_path); labels.append(label_map[patient_dir])
                except Exception as e:
                    print(f"[WARN] {dcm_path}: {e}")

    keep = [i for i,l in enumerate(labels) if l in (0,1)]
    image_paths = [image_paths[i] for i in keep]; labels = [labels[i] for i in keep]

    tr_p, te_p, tr_y, te_y = train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=42)
    tr_p, va_p, tr_y, va_y = train_test_split(tr_p, tr_y, test_size=0.125, stratify=tr_y, random_state=42)

    for split, (paths, ys) in {'train': (tr_p,tr_y),'val':(va_p,va_y),'test':(te_p,te_y)}.items():
        split_dir = os.path.join(output_dir, split); os.makedirs(split_dir, exist_ok=True)
        pd.DataFrame({'image_path': paths, 'label': ys}).to_csv(os.path.join(split_dir, f'{split}_labels.csv'), index=False)
    print(f"[OK] CMMD processed at: {output_dir}")
