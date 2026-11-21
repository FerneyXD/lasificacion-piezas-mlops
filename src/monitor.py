"""Etapa de monitoreo: deteccion de drift con Evidently y registro en MLflow.

POR COMPLETAR. Consulte la Fase 4 de la guia.

Ya dispone de: image_features.image_features / FEATURES, common.*, model_utils.load_model,
list_images en common, y la seccion `monitor` de params.yaml.

Uso esperado:
  python src/monitor.py                                        # referencia = train, actual = test
  python src/monitor.py --current-dir data/production/batch_02 # actual = un lote de produccion

Salidas OBLIGATORIAS:
  reports/drift_report.html   reporte de Evidently
  reports/drift.json          metricas resumen de drift (rastreadas por DVC)
Registre ademas parametros, metricas y el reporte en MLflow (experimento de params['monitor']).
"""
import argparse

from common import list_images, load_params, read_json, setup_mlflow, write_json  # noqa: F401
from image_features import FEATURES, image_features  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-dir", default="data/processed/test")
    args = ap.parse_args()  # noqa: F841

    # TODO 1: tomar una muestra (params['monitor']['sample_size']) de data/processed/train (referencia)
    #         y de args.current_dir (actual)
    # TODO 2: construir un DataFrame por conjunto con las features de imagen; si
    #         params['monitor']['with_model'] es true, agregar columnas `prediction` y `confidence`
    # TODO 3: si params['monitor']['simulate_drift'] es true, degradar las imagenes actuales
    #         (por ejemplo oscurecer y desenfocar) antes de extraer features y predecir
    # TODO 4: generar el reporte de drift con Evidently (version fijada en requirements.txt)
    # TODO 5: extraer del reporte la proporcion de columnas con drift y si hay drift de dataset;
    #         escribir reports/drift.json y registrar todo en MLflow
    raise NotImplementedError


if __name__ == "__main__":
    main()
