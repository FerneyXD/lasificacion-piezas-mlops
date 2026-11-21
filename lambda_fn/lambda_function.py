"""Lambda disparada por S3 que clasifica y mueve imagenes.

Variables de entorno (se configuran en la Lambda):
  ENDPOINT_NAME     nombre del endpoint de SageMaker
  CLASS_NAMES       clases separadas por coma, en el orden de models/classes.json
  THRESHOLD         umbral de confianza (por defecto 0.80)
  INCOMING_PREFIX   prefijo de entrada (por defecto 'entrada/')
"""
import json
import os
import posixpath
import urllib.parse

import boto3

s3 = boto3.client("s3")
smr = boto3.client("sagemaker-runtime")

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
CLASS_NAMES = [c.strip() for c in os.environ["CLASS_NAMES"].split(",")]
THRESHOLD = float(os.environ.get("THRESHOLD", "0.80"))
INCOMING_PREFIX = os.environ.get("INCOMING_PREFIX", "entrada/")
OUTPUT_PREFIX = "clasificadas/"
UNKNOWN_PREFIX = "no-identificadas/"

CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
}


def predict(image_bytes: bytes, content_type: str):
    """Invoca el endpoint de SageMaker y devuelve (clase_predicha, probabilidad)."""
    resp = smr.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType=content_type,
        Accept="application/json",
        Body=image_bytes,
    )
    payload = json.loads(resp["Body"].read().decode("utf-8"))
    probs = payload["predictions"][0]
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return CLASS_NAMES[best_idx], float(probs[best_idx])


def resolve_destination(filename: str, label: str, prob: float) -> str:
    """Determina la clave de destino en S3 aplicando la regla estricta prob > THRESHOLD."""
    if prob > THRESHOLD:
        return f"{OUTPUT_PREFIX}{label}/{filename}"
    return f"{UNKNOWN_PREFIX}{filename}"


def lambda_handler(event, context):
    """Recorre event['Records'], clasifica imagenes en INCOMING_PREFIX y las mueve."""
    processed = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        raw_key = record["s3"]["object"]["key"]
        key = urllib.parse.unquote_plus(raw_key)

        if not key.startswith(INCOMING_PREFIX) or key.endswith("/"):
            continue
        ext = posixpath.splitext(key)[1].lower()
        content_type = CONTENT_TYPES.get(ext)
        if not content_type:
            continue

        obj = s3.get_object(Bucket=bucket, Key=key)
        image_bytes = obj["Body"].read()
        label, prob = predict(image_bytes, content_type)

        filename = posixpath.basename(key)
        dest = resolve_destination(filename, label, prob)
        s3.copy_object(Bucket=bucket, Key=dest, CopySource={"Bucket": bucket, "Key": key})
        s3.delete_object(Bucket=bucket, Key=key)

        processed.append({"key": key, "dest": dest, "label": label, "prob": prob})

    return {"processed": processed}
