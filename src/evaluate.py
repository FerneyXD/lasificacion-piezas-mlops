"""Etapa de evaluacion: metricas en test y analisis del umbral de incertidumbre.

Salidas generadas:
  reports/metrics.json            metricas de test (rastreadas por DVC)
  reports/confusion_matrix.png    matriz de confusion
  reports/threshold_curve.csv     columnas: threshold, coverage, accuracy_accepted
Registra ademas metricas y graficos en la misma corrida de MLflow del entrenamiento.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from common import load_params, read_json, setup_mlflow, write_json
from model_utils import get_dataset, load_model

THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]


def extract_labels_and_predictions(model, test_ds):
    """Obtiene etiquetas reales y probabilidades predichas respetando el orden del dataset."""
    y_true_batches = [np.argmax(y.numpy(), axis=1) for _, y in test_ds]
    y_true = np.concatenate(y_true_batches, axis=0)
    y_prob = model.predict(test_ds, verbose=0)
    if len(y_true) != len(y_prob):
        raise RuntimeError(f"Desajuste de muestras: {len(y_true)} etiquetas vs {len(y_prob)} predicciones")
    return y_true, y_prob


def save_confusion_matrix(y_true, y_pred, class_names: list, out_path: Path) -> None:
    """Dibuja y guarda la matriz de confusion en formato PNG."""
    cm = confusion_matrix(y_true, y_pred)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_ylabel("Clase real")
    ax.set_xlabel("Clase predicha")
    ax.set_title("Matriz de confusion (test)")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_at_threshold(y_true, y_pred, conf, threshold: float) -> tuple[float, float, int, int]:
    """Calcula cobertura y accuracy aceptada usando regla estricta (conf > threshold)."""
    accepted = conf > threshold
    n_accepted = int(np.sum(accepted))
    n_rejected = int(len(conf) - n_accepted)
    coverage = float(n_accepted / len(conf)) if len(conf) else 0.0
    acc_accepted = float(np.mean(y_pred[accepted] == y_true[accepted])) if n_accepted > 0 else 0.0
    return coverage, acc_accepted, n_accepted, n_rejected


def write_threshold_curve(y_true, y_pred, conf, out_path: Path) -> list[dict]:
    """Calcula la curva cobertura vs accuracy aceptada y la guarda en CSV."""
    rows = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["threshold", "coverage", "accuracy_accepted"])
        for thr in THRESHOLDS:
            cov, acc_acc, _, _ = evaluate_at_threshold(y_true, y_pred, conf, thr)
            row = {"threshold": thr, "coverage": round(cov, 6), "accuracy_accepted": round(acc_acc, 6)}
            rows.append(row)
            writer.writerow([f"{thr:.2f}", f"{cov:.6f}", f"{acc_acc:.6f}"])
    return rows


def main() -> None:
    params = load_params()
    data_cfg = params["data"]
    train_cfg = params["train"]
    threshold = float(params["evaluate"]["threshold"])

    class_names = read_json("models/classes.json")["class_names"]
    run_id = read_json("models/train_info.json")["run_id"]
    model = load_model("models/model.keras")

    test_ds = get_dataset(
        Path(data_cfg["processed_dir"]) / "test",
        int(train_cfg["img_size"]),
        int(train_cfg["batch_size"]),
        shuffle=False,
        class_names=class_names,
    ).prefetch(1)

    y_true, y_prob = extract_labels_and_predictions(model, test_ds)
    y_pred = np.argmax(y_prob, axis=1)
    conf = np.max(y_prob, axis=1)

    test_acc = float(accuracy_score(y_true, y_pred))
    test_f1 = float(f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4), flush=True)

    cm_path = Path("reports/confusion_matrix.png")
    curve_path = Path("reports/threshold_curve.csv")
    save_confusion_matrix(y_true, y_pred, class_names, cm_path)
    write_threshold_curve(y_true, y_pred, conf, curve_path)

    cov, acc_acc, n_acc, n_rej = evaluate_at_threshold(y_true, y_pred, conf, threshold)
    metrics = {
        "test_accuracy": round(test_acc, 6),
        "test_f1_macro": round(test_f1, 6),
        "threshold": threshold,
        "coverage": round(cov, 6),
        "accuracy_accepted": round(acc_acc, 6),
        "mean_confidence": round(float(np.mean(conf)), 6),
        "n_test": int(len(y_true)),
        "n_accepted": n_acc,
        "n_rejected": n_rej,
    }
    write_json("reports/metrics.json", metrics)
    print(f"Evaluacion test: {metrics}", flush=True)

    mlflow = setup_mlflow(train_cfg["experiment_name"])
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("evaluate.threshold", str(threshold))
        mlflow.log_metrics({
            "test_accuracy": metrics["test_accuracy"],
            "test_f1_macro": metrics["test_f1_macro"],
            "eval_threshold": threshold,
            "eval_coverage": metrics["coverage"],
            "eval_accuracy_accepted": metrics["accuracy_accepted"],
            "eval_mean_confidence": metrics["mean_confidence"],
        })
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact("reports/metrics.json")


if __name__ == "__main__":
    main()
