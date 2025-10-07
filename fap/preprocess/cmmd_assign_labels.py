import os
import pandas as pd

def abspath(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))

def assign_labels_to_cmmd_images(cmmd_dir, excel_path, output_csv):
    cmmd_dir = abspath(cmmd_dir); excel_path = abspath(excel_path); output_csv = abspath(output_csv)
    df = pd.read_excel(excel_path, dtype={"ID1": str, "classification": str})
    label_map = {}
    for _, row in df.iterrows():
        if row['ID1'] not in label_map:
            label_map[row['ID1']] = 0 if str(row['classification']).strip().lower() == "benign" else 1
    image_paths, labels = [], []
    for patient_folder in os.listdir(cmmd_dir):
        patient_path = os.path.join(cmmd_dir, patient_folder)
        if not os.path.isdir(patient_path) or not patient_folder.startswith("D"):
            continue
        if patient_folder not in label_map:
            print(f"Warning: PatientID {patient_folder} not found in Excel. Skipping."); continue
        label = label_map[patient_folder]
        for file in os.listdir(patient_path):
            if file.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff','.tif','.webp')):
                img_path = abspath(os.path.join(patient_path, file))
                image_paths.append(img_path); labels.append(label)
    pd.DataFrame({'image_path': image_paths, 'label': labels}).to_csv(output_csv, index=False)
    print(f"Saved {len(image_paths)} entries to {output_csv}")
