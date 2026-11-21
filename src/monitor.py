"""Etapa de monitoreo: deteccion de drift con Evidently y registro en MLflow.

Salidas generadas:
  reports/drift_report.html   reporte de Evidently
  reports/drift.json          metricas resumen de drift (rastreadas por DVC)
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from PIL import Image, ImageEnhance, ImageFilter

from common import list_images, load_params, read_json, setup_mlflow, write_json
from image_features import FEATURES, image_features
from model_utils import load_model

SIM_BRIGHTNESS = 0.5
SIM_BLUR_RADIUS = 2.0


def sample_image_paths(folder: str, sample_size: int, seed: int) -> list[Path]:
    """Selecciona una muestra reproducible de rutas de imagen en una carpeta."""
    files = list_images(folder)
    if not files:
        raise SystemExit(f"Sin imagenes en {folder}")
    rng = random.Random(seed)
    return rng.sample(files, min(sample_size, len(files)))


def degrade_image(img: Image.Image) -> Image.Image:
    """Aplica degradacion sintetica (oscuridad + desenfoque) para simular drift."""
    dark = ImageEnhance.Brightness(img).enhance(SIM_BRIGHTNESS)
    return dark.filter(ImageFilter.GaussianBlur(SIM_BLUR_RADIUS))


def build_feature_dataframe(
    paths: list[Path],
    img_size: int,
    with_model: bool,
    model=None,
    class_names: list[str] | None = None,
    apply_drift: bool = False,
) -> pd.DataFrame:
    """Extrae features de imagen y, opcionalmente, prediccion y confianza del modelo."""
    rows = []
    batch_arrays = []
    for p in paths:
        with Image.open(p) as raw_img:
            img = raw_img.convert("RGB")
            if apply_drift:
                img = degrade_image(img)
            rows.append(image_features(img))
            if with_model:
                resized = img.resize((img_size, img_size), Image.BILINEAR)
                batch_arrays.append(np.asarray(resized, dtype=np.float32))

    df = pd.DataFrame(rows, columns=FEATURES)
    if with_model:
        if model is None or not class_names:
            raise ValueError("Se requiere modelo y class_names cuando with_model=true")
        arr = np.stack(batch_arrays, axis=0)
        probs = model.predict(arr, batch_size=32, verbose=0)
        indices = np.argmax(probs, axis=1)
        df["prediction"] = [class_names[i] for i in indices]
        df["confidence"] = np.max(probs, axis=1).astype(float)
    return df


def extract_drift_summary(report_dict: dict) -> dict:
    """Extrae metricas agregadas y detalle por columna del diccionario de Evidently 0.4."""
    dataset_drift = False
    share_drifted = 0.0
    n_drifted = 0
    n_cols = 0
    drifted_columns = []

    for metric in report_dict.get("metrics", []):
        res = metric.get("result", {})
        if "share_of_drifted_columns" in res and "dataset_drift" in res:
            dataset_drift = bool(res["dataset_drift"])
            share_drifted = float(res["share_of_drifted_columns"])
            n_drifted = int(res.get("number_of_drifted_columns", 0))
            n_cols = int(res.get("number_of_columns", 0))
        if "drift_by_columns" in res:
            for col, info in res["drift_by_columns"].items():
                if info.get("drift_detected"):
                    drifted_columns.append(col)

    return {
        "dataset_drift": dataset_drift,
        "share_of_drifted_columns": round(share_drifted, 6),
        "number_of_drifted_columns": n_drifted,
        "number_of_columns": n_cols,
        "drifted_columns": sorted(set(drifted_columns)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-dir", default="data/processed/test")
    args = ap.parse_args()

    params = load_params()
    data_cfg = params["data"]
    mon_cfg = params["monitor"]
    img_size = int(params["train"]["img_size"])
    threshold = float(params["evaluate"]["threshold"])
    seed = int(data_cfg["seed"])

    sample_size = int(mon_cfg["sample_size"])
    with_model = bool(mon_cfg["with_model"])
    simulate_drift = bool(mon_cfg["simulate_drift"])

    ref_dir = str(Path(data_cfg["processed_dir"]) / "train")
    ref_paths = sample_image_paths(ref_dir, sample_size, seed)
    cur_paths = sample_image_paths(args.current_dir, sample_size, seed + 1)

    model = load_model("models/model.keras") if with_model else None
    class_names = read_json("models/classes.json")["class_names"] if with_model else None

    ref_df = build_feature_dataframe(ref_paths, img_size, with_model, model, class_names, False)
    cur_df = build_feature_dataframe(cur_paths, img_size, with_model, model, class_names, simulate_drift)

    num_cols = list(FEATURES) + (["confidence"] if with_model else [])
    cat_cols = ["prediction"] if with_model else []
    mapping = ColumnMapping(numerical_features=num_cols, categorical_features=cat_cols)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=mapping)

    Path("reports").mkdir(parents=True, exist_ok=True)
    html_path = "reports/drift_report.html"
    report.save_html(html_path)

    summary = extract_drift_summary(report.as_dict())
    summary.update({
        "reference_dir": ref_dir,
        "current_dir": args.current_dir,
        "n_reference": len(ref_df),
        "n_current": len(cur_df),
        "simulate_drift": simulate_drift,
        "with_model": with_model,
    })

    if with_model:
        ref_unknown = ref_df["confidence"] <= threshold
        cur_unknown = cur_df["confidence"] <= threshold
        summary.update({
            "threshold": threshold,
            "mean_confidence_ref": round(float(ref_df["confidence"].mean()), 6),
            "mean_confidence_cur": round(float(cur_df["confidence"].mean()), 6),
            "n_unknown_cur": int(cur_unknown.sum()),
            "pct_unknown_ref": round(float(ref_unknown.mean()), 6),
            "pct_unknown_cur": round(float(cur_unknown.mean()), 6),
        })

    json_path = "reports/drift.json"
    write_json(json_path, summary)
    print(f"Monitoreo drift ({args.current_dir}): {summary}", flush=True)

    mlflow = setup_mlflow(mon_cfg["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params({
            "current_dir": args.current_dir,
            "sample_size": sample_size,
            "simulate_drift": simulate_drift,
            "with_model": with_model,
            "threshold": threshold,
        })
        numeric_metrics = {
            "share_of_drifted_columns": summary["share_of_drifted_columns"],
            "dataset_drift": int(summary["dataset_drift"]),
            "number_of_drifted_columns": summary["number_of_drifted_columns"],
            "n_current": summary["n_current"],
        }
        if with_model:
            numeric_metrics.update({
                "mean_confidence_ref": summary["mean_confidence_ref"],
                "mean_confidence_cur": summary["mean_confidence_cur"],
                "pct_unknown_cur": summary["pct_unknown_cur"],
                "n_unknown_cur": summary["n_unknown_cur"],
            })
        mlflow.log_metrics(numeric_metrics)
        mlflow.log_artifact(html_path)
        mlflow.log_artifact(json_path)


if __name__ == "__main__":
    main()
