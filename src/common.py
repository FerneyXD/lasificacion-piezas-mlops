"""Utilidades livianas (sin TensorFlow) compartidas por todas las etapas."""
import json
import os
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_mlflow(experiment: str):
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(experiment)
    return mlflow


def write_json(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def list_images(folder: str):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)
