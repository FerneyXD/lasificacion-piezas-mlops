"""Lambda disparada por S3 que clasifica y mueve imagenes.

POR COMPLETAR. Consulte la Fase 6 de la guia.

Variables de entorno (se configuran en la Lambda):
  ENDPOINT_NAME     nombre del endpoint de SageMaker
  CLASS_NAMES       clases separadas por coma, en el orden de models/classes.json
  THRESHOLD         umbral de confianza (por defecto 0.80)
  INCOMING_PREFIX   prefijo de entrada (por defecto 'entrada/')

Reglas de negocio:
  - Solo se procesan objetos cuya clave empiece por INCOMING_PREFIX (y no terminen en '/').
  - Solo extensiones .jpg .jpeg .png .bmp; el resto se ignora.
  - Si la probabilidad maxima es ESTRICTAMENTE mayor que THRESHOLD, el archivo se mueve a
        clasificadas/<clase>/<nombre_archivo>
    si no, a
        no-identificadas/<nombre_archivo>
  - "Mover" en S3 = copiar al destino y borrar el original.
  - Las claves llegan codificadas en URL (espacios como '+', etc.): decodificarlas.

Contrato con las pruebas (tests/test_lambda.py):
  - Los clientes deben llamarse `s3` y `smr` (nivel de modulo).
  - lambda_handler(event, context) devuelve {"processed": [{"key", "dest", "label", "prob"}, ...]}
    con una entrada por cada imagen procesada (las ignoradas no aparecen).
"""
import json  # noqa: F401
import os
import urllib.parse  # noqa: F401

import boto3

s3 = boto3.client("s3")
smr = boto3.client("sagemaker-runtime")

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
CLASS_NAMES = [c.strip() for c in os.environ["CLASS_NAMES"].split(",")]
THRESHOLD = float(os.environ.get("THRESHOLD", "0.80"))
INCOMING_PREFIX = os.environ.get("INCOMING_PREFIX", "entrada/")
OUTPUT_PREFIX = "clasificadas/"
UNKNOWN_PREFIX = "no-identificadas/"


def predict(image_bytes: bytes, content_type: str):
    """TODO: invocar el endpoint y devolver (clase_predicha, probabilidad)."""
    raise NotImplementedError


def lambda_handler(event, context):
    """TODO: recorrer event['Records'], aplicar las reglas de negocio y devolver el resumen."""
    raise NotImplementedError
