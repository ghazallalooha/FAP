import os, yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

_CFG_ENV = "FAP_DATASETS_YAML"

@dataclass
class DatasetPaths:
    cmmd_dicom: Optional[str] = None
    cmmd_excel: Optional[str] = None
    cmmd_processed: Optional[str] = None
    breast_images: Optional[str] = None
    breast_csv: Optional[str] = None
    cbis_dicom: Optional[str] = None
    cbis_labels_csv: Optional[str] = None

def _expand(p: Optional[str]) -> Optional[str]:
    if not p: return None
    return str(Path(os.path.expanduser(os.path.expandvars(p))).resolve())

def load_yaml(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = os.getenv(_CFG_ENV)
    if path is None:
        maybe = Path("configs/datasets.yaml")
        if maybe.exists():
            path = str(maybe)
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_paths(yaml_path: Optional[str] = None) -> DatasetPaths:
    cfg = load_yaml(yaml_path)
    d = cfg.get("datasets", {})
    return DatasetPaths(
        cmmd_dicom=_expand(d.get("cmmd_dicom")),
        cmmd_excel=_expand(d.get("cmmd_excel")),
        cmmd_processed=_expand(d.get("cmmd_processed")),
        breast_images=_expand(d.get("breast_images")),
        breast_csv=_expand(d.get("breast_csv")),
        cbis_dicom=_expand(d.get("cbis_dicom")),
        cbis_labels_csv=_expand(d.get("cbis_labels_csv")),
    )
