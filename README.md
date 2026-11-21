# Código base: clasificación de piezas industriales

Este repositorio es el punto de partida del proyecto. **Todas las instrucciones, requisitos,
contratos entre componentes y entregables están en la guía en PDF.** Aquí solo se resume qué trae.

## Qué incluye

| Ruta | Estado |
|---|---|
| `params.yaml`, `requirements.txt`, `.gitignore` | Listos |
| `src/common.py`, `src/model_utils.py`, `src/image_features.py` | Ayudantes listos |
| `src/prepare.py` | Listo (división estratificada train/val/test) |
| `src/simulate_production.py` | Listo (genera lotes de "producción" degradados) |
| `dvc.yaml` | Solo la etapa `prepare`; falta el resto |
| `src/train.py`, `src/evaluate.py`, `src/monitor.py` | **Por completar** |
| `src/deploy_sagemaker.py`, `sagemaker_code/inference.py` | **Por completar** |
| `lambda_fn/lambda_function.py` | **Por completar** |
| `tests/test_lambda.py` | Pruebas de aceptación de la Lambda (no requieren AWS) |

Busque los comentarios `TODO` y las cadenas `NotImplementedError` para ver lo que falta.

## Primeros pasos

```bash
python3.10 -m venv .venv && source .venv/bin/activate   # Python 3.10 u 3.11
pip install -r requirements.txt
```

Verificación de la Lambda sin AWS: `python -m unittest tests/test_lambda.py -v`
