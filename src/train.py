"""Etapa de entrenamiento: transfer learning con VGG16 (ImageNet) + tracking en MLflow.

POR COMPLETAR. Consulte la Fase 1 de la guia.

Ya dispone de:
  - common.load_params / common.setup_mlflow / common.write_json
  - model_utils.VGGPreprocess  (capa de preprocesamiento VGG16 para incluir dentro del modelo)
  - model_utils.get_dataset    (lee data/processed/<split>/<clase>/*.jpg como dataset de Keras)
  - Hiperparametros en params.yaml (seccion `train`)

Salidas OBLIGATORIAS (otras etapas y los ayudantes dependen de estos nombres):
  models/model.keras                    modelo completo (lo carga model_utils.load_model)
  models/saved_model/1/                 SavedModel en formato TF, listo para TF Serving / SageMaker
  models/classes.json                   {"class_names": [...]}  en el orden de salida del modelo
  models/train_info.json                {"run_id": "<id de la corrida de MLflow>"}
  reports/train_metrics.json            metricas de validacion finales (rastreadas por DVC)
"""
from common import load_params, setup_mlflow, write_json  # noqa: F401
from model_utils import VGGPreprocess, get_dataset  # noqa: F401


def build_model(n_classes: int, img_size: int, dense_units: int, dropout: float):
    """TODO: construir el modelo (base VGG16 + cabeza de clasificacion nueva).

    Debe devolver (modelo, base_vgg16) para poder descongelar capas despues.
    """
    raise NotImplementedError


def main():
    # TODO 1: cargar params y configurar MLflow (experimento params['train']['experiment_name'])
    # TODO 2: crear datasets train/val
    # TODO 3: construir el modelo y entrenar en dos fases (cabeza, luego fine-tuning de las ultimas capas)
    # TODO 4: registrar en MLflow parametros, metricas por epoca, y artefactos
    #         (classes.json, SavedModel, params.yaml y, si existen, data/raw.dvc y dvc.lock)
    # TODO 5: guardar los archivos de salida obligatorios (ver docstring del modulo)
    raise NotImplementedError


if __name__ == "__main__":
    main()
