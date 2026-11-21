"""Etapa de entrenamiento: transfer learning con VGG16 (ImageNet) + tracking en MLflow.

Salidas generadas:
  models/model.keras                    modelo completo (lo carga model_utils.load_model)
  models/saved_model/1/                 SavedModel en formato TF, listo para TF Serving / SageMaker
  models/classes.json                   {"class_names": [...]}  en el orden de salida del modelo
  models/train_info.json                {"run_id": "<id de la corrida de MLflow>"}
  reports/train_metrics.json            metricas de validacion finales (rastreadas por DVC)
"""
import shutil
import time
from pathlib import Path

import tensorflow as tf

from common import load_params, setup_mlflow, write_json
from model_utils import VGGPreprocess, get_dataset

AUGMENT_ROTATION_FACTOR = 0.08
AUGMENT_ZOOM_FACTOR = 0.10


class EpochLoggerCallback(tf.keras.callbacks.Callback):
    """Registra metricas por epoca en MLflow e imprime progreso en tiempo real."""

    def __init__(self, mlflow_client, phase_name: str, epoch_offset: int, total_epochs: int):
        super().__init__()
        self._mlflow = mlflow_client
        self._phase_name = phase_name
        self._epoch_offset = epoch_offset
        self._total_epochs = total_epochs
        self._epoch_start = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.perf_counter() - self._epoch_start
        global_step = self._epoch_offset + epoch + 1
        metrics = {
            "train_loss": float(logs.get("loss", 0.0)),
            "train_accuracy": float(logs.get("accuracy", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
            "epoch_seconds": float(elapsed),
        }
        self._mlflow.log_metrics(metrics, step=global_step)
        print(
            f"[{self._phase_name}] Epoca {global_step}/{self._total_epochs} | "
            f"loss={metrics['train_loss']:.4f} acc={metrics['train_accuracy']:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_accuracy']:.4f} | "
            f"{elapsed:.1f}s",
            flush=True,
        )


def configure_runtime(seed: int) -> bool:
    """Fija semillas y configura memoria GPU + precision mixta si hay GPU disponible."""
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print(f"GPU detectada ({len(gpus)}): precision mixta activada.", flush=True)
        return True
    print("Sin GPU detectada: ejecutando en CPU float32.", flush=True)
    return False


def build_model(n_classes: int, img_size: int, dense_units: int, dropout: float):
    """Construye el modelo con aumento de datos, VGGPreprocess, base VGG16 y cabeza nueva.

    Devuelve (modelo, base_vgg16) para poder descongelar capas en fine-tuning.
    """
    if n_classes < 2:
        raise ValueError(f"Se requieren al menos 2 clases para clasificar; recibido {n_classes}.")
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image_input")
    x = tf.keras.layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = tf.keras.layers.RandomRotation(AUGMENT_ROTATION_FACTOR, name="aug_rot")(x)
    x = tf.keras.layers.RandomZoom(AUGMENT_ZOOM_FACTOR, name="aug_zoom")(x)
    x = VGGPreprocess(name="vgg_preprocess")(x)

    base = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        n_classes, activation="softmax", dtype="float32", name="predictions"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="industrial_vgg16")
    return model, base


def unfreeze_top_layers(base_model: tf.keras.Model, unfreeze_last_n: int) -> int:
    """Descongela las ultimas N capas de la base VGG16 manteniendo el resto congelado."""
    if unfreeze_last_n <= 0:
        base_model.trainable = False
        return 0
    base_model.trainable = True
    cutoff = max(0, len(base_model.layers) - unfreeze_last_n)
    for layer in base_model.layers[:cutoff]:
        layer.trainable = False
    for layer in base_model.layers[cutoff:]:
        layer.trainable = True
    return len(base_model.layers) - cutoff


def export_models_fp32(
    trained_model: tf.keras.Model,
    n_classes: int,
    img_size: int,
    dense_units: int,
    dropout: float,
) -> None:
    """Exporta model.keras y saved_model/1/ en float32 para compatibilidad train/serve."""
    tf.keras.mixed_precision.set_global_policy("float32")
    export_model, _ = build_model(n_classes, img_size, dense_units, dropout)
    export_model.set_weights(trained_model.get_weights())
    export_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    Path("models").mkdir(parents=True, exist_ok=True)
    export_model.save("models/model.keras")

    saved_dir = Path("models/saved_model/1")
    if saved_dir.exists():
        shutil.rmtree(saved_dir)
    saved_dir.parent.mkdir(parents=True, exist_ok=True)
    export_model.save(str(saved_dir), save_format="tf")


def log_lineage_artifacts(mlflow_client) -> None:
    """Registra artefactos de configuracion, clases, SavedModel y linaje DVC si existen."""
    mlflow_client.log_artifact("models/classes.json")
    mlflow_client.log_artifact("params.yaml")
    mlflow_client.log_artifacts("models/saved_model/1", artifact_path="saved_model/1")
    for lineage_path in ("data/raw.dvc", "dvc.lock"):
        if Path(lineage_path).exists():
            mlflow_client.log_artifact(lineage_path, artifact_path="lineage")


def run_phase(
    model: tf.keras.Model,
    train_ds,
    val_ds,
    lr: float,
    epochs: int,
    mlflow_client,
    phase_name: str,
    epoch_offset: int,
    total_epochs: int,
):
    """Compila y entrena una fase del modelo retornando el historial."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callback = EpochLoggerCallback(mlflow_client, phase_name, epoch_offset, total_epochs)
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[callback],
        verbose=0,
    )


def main() -> None:
    params = load_params()
    data_cfg = params["data"]
    train_cfg = params["train"]
    seed = int(data_cfg["seed"])

    configure_runtime(seed)

    processed_dir = Path(data_cfg["processed_dir"])
    img_size = int(train_cfg["img_size"])
    batch_size = int(train_cfg["batch_size"])
    epochs_head = int(train_cfg["epochs_head"])
    epochs_finetune = int(train_cfg["epochs_finetune"])
    total_epochs = epochs_head + epochs_finetune

    train_raw = get_dataset(processed_dir / "train", img_size, batch_size, shuffle=True)
    class_names = list(train_raw.class_names)
    val_raw = get_dataset(
        processed_dir / "val", img_size, batch_size, shuffle=False, class_names=class_names
    )

    n_train = int(tf.data.experimental.cardinality(train_raw).numpy()) * batch_size
    n_val = int(tf.data.experimental.cardinality(val_raw).numpy()) * batch_size
    train_ds = train_raw.prefetch(tf.data.AUTOTUNE)
    val_ds = val_raw.prefetch(tf.data.AUTOTUNE)

    mlflow = setup_mlflow(train_cfg["experiment_name"])
    model, base = build_model(
        n_classes=len(class_names),
        img_size=img_size,
        dense_units=int(train_cfg["dense_units"]),
        dropout=float(train_cfg["dropout"]),
    )

    with mlflow.start_run() as run:
        mlflow.log_params({**train_cfg, "seed": seed, "n_classes": len(class_names)})
        mlflow.log_params({"approx_train_samples": n_train, "approx_val_samples": n_val})

        print(
            f"Iniciando entrenamiento MLflow run_id={run.info.run_id} "
            f"(clases={len(class_names)}, epocas={epochs_head}+{epochs_finetune})",
            flush=True,
        )
        history = run_phase(
            model, train_ds, val_ds,
            lr=float(train_cfg["lr_head"]),
            epochs=epochs_head,
            mlflow_client=mlflow,
            phase_name="Cabeza",
            epoch_offset=0,
            total_epochs=total_epochs,
        )

        if epochs_finetune > 0:
            unfrozen = unfreeze_top_layers(base, int(train_cfg["unfreeze_last_n"]))
            print(f"Descongeladas {unfrozen} capas finales de VGG16 para fine-tuning.", flush=True)
            history = run_phase(
                model, train_ds, val_ds,
                lr=float(train_cfg["lr_finetune"]),
                epochs=epochs_finetune,
                mlflow_client=mlflow,
                phase_name="FineTune",
                epoch_offset=epochs_head,
                total_epochs=total_epochs,
            )

        final_metrics = {
            "val_loss": float(history.history["val_loss"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
            "train_loss": float(history.history["loss"][-1]),
            "train_accuracy": float(history.history["accuracy"][-1]),
            "epochs_total": total_epochs,
        }
        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items() if k != "epochs_total"})

        export_models_fp32(
            model,
            n_classes=len(class_names),
            img_size=img_size,
            dense_units=int(train_cfg["dense_units"]),
            dropout=float(train_cfg["dropout"]),
        )
        write_json("models/classes.json", {"class_names": class_names})
        write_json("models/train_info.json", {"run_id": run.info.run_id})
        write_json("reports/train_metrics.json", final_metrics)
        log_lineage_artifacts(mlflow)

        print(f"Entrenamiento finalizado. Metricas finales: {final_metrics}", flush=True)


if __name__ == "__main__":
    main()
