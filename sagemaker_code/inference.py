"""Handlers de pre/post-procesamiento para el contenedor TensorFlow Serving de SageMaker.

Contrato del endpoint:
  Entrada : bytes de una imagen con Content-Type image/jpeg o image/png
  Salida  : JSON {"predictions": [[p_clase_0, p_clase_1, ...]]}
"""
import io
import json

import numpy as np
from PIL import Image

TARGET_SIZE = (224, 224)
SUPPORTED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "application/octet-stream"}


def input_handler(data, context):
    """Convierte bytes de imagen a cuerpo JSON {"instances": [...]} para TF Serving."""
    content_type = (getattr(context, "request_content_type", "") or "").split(";")[0].strip().lower()
    if content_type not in SUPPORTED_TYPES:
        raise ValueError(f"Content-Type no soportado: {content_type}")

    raw_bytes = data.read() if hasattr(data, "read") else data
    with Image.open(io.BytesIO(raw_bytes)) as img:
        rgb = img.convert("RGB").resize(TARGET_SIZE, Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32)

    return json.dumps({"instances": [arr.tolist()]})


def output_handler(response, context):
    """Valida la respuesta de TF Serving y retorna (contenido, accept_header)."""
    if response.status_code != 200:
        body = response.content.decode("utf-8", errors="replace")
        raise ValueError(f"Error de TF Serving ({response.status_code}): {body}")
    accept = getattr(context, "accept_header", None) or "application/json"
    return response.content, accept
