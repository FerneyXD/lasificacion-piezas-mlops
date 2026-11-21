"""Etapa de evaluacion: metricas en test y analisis del umbral de incertidumbre.

POR COMPLETAR. Consulte la Fase 2 de la guia.

Ya dispone de: common.*, model_utils.get_dataset, model_utils.load_model,
y el umbral inicial en params.yaml (seccion `evaluate`).

Salidas OBLIGATORIAS:
  reports/metrics.json            metricas de test (rastreadas por DVC)
  reports/confusion_matrix.png    matriz de confusion
  reports/threshold_curve.csv     columnas: threshold, coverage, accuracy_accepted
Ademas, registre estos resultados en la MISMA corrida de MLflow del entrenamiento
(el run_id esta en models/train_info.json).
"""
from common import load_params, read_json, setup_mlflow, write_json  # noqa: F401
from model_utils import get_dataset, load_model  # noqa: F401


def main():
    # TODO 1: cargar modelo, nombres de clases y dataset de test (sin mezclar el orden)
    # TODO 2: predecir; calcular accuracy y F1 macro; imprimir classification_report
    # TODO 3: dibujar y guardar la matriz de confusion
    # TODO 4: para varios umbrales (0.50 ... 0.95) calcular cobertura (fraccion de imagenes aceptadas,
    #         es decir, con confianza ESTRICTAMENTE mayor al umbral) y accuracy sobre las aceptadas
    # TODO 5: calcular las metricas al umbral de params.yaml y escribir reports/metrics.json
    # TODO 6: registrar metricas y graficos en MLflow (corrida existente)
    raise NotImplementedError


if __name__ == "__main__":
    main()
