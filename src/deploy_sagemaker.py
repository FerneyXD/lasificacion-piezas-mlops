"""Despliegue del modelo en un endpoint de inferencia en tiempo real de Amazon SageMaker.

Uso esperado:
  python src/deploy_sagemaker.py --role arn:aws:iam::<cuenta>:role/<rol> --bucket <bucket>
  python src/deploy_sagemaker.py --delete --endpoint-name industrial-vgg16

Salida OBLIGATORIA al desplegar: reports/deploy.json con
  {"endpoint_name": ..., "model_data": ..., "CLASS_NAMES": "clase1,clase2,..."}
"""
import argparse
import tarfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

from common import read_json, write_json

SAVED_MODEL_DIR = Path("models/saved_model/1")
PACKAGE_PATH = Path("models/model.tar.gz")
CLASSES_PATH = Path("models/classes.json")
DEPLOY_REPORT_PATH = "reports/deploy.json"


def delete_endpoint(endpoint_name: str) -> None:
    """Elimina el endpoint y su endpoint-config si existen en SageMaker."""
    sm = boto3.client("sagemaker")
    try:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        config_name = desc.get("EndpointConfigName")
        print(f"Eliminando endpoint '{endpoint_name}'...", flush=True)
        sm.delete_endpoint(EndpointName=endpoint_name)
        waiter = sm.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=endpoint_name)
        print(f"Endpoint '{endpoint_name}' eliminado.", flush=True)
    except ClientError as exc:
        if "Could not find endpoint" not in str(exc):
            raise
        print(f"Endpoint '{endpoint_name}' no existe.", flush=True)
        config_name = endpoint_name

    for cfg in {c for c in (config_name, endpoint_name) if c}:
        try:
            sm.delete_endpoint_config(EndpointConfigName=cfg)
            print(f"EndpointConfig '{cfg}' eliminado.", flush=True)
        except ClientError as exc:
            if "Could not find endpoint configuration" not in str(exc):
                raise


def package_saved_model(saved_model_dir: Path, archive_path: Path) -> Path:
    """Empaqueta models/saved_model/1/ en model.tar.gz con la carpeta 1/ en la raiz."""
    if not (saved_model_dir / "saved_model.pb").exists():
        raise FileNotFoundError(f"No existe SavedModel en {saved_model_dir}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(saved_model_dir, arcname="1")
    return archive_path


def deploy_endpoint(role: str, bucket: str, endpoint_name: str, instance_type: str) -> None:
    """Empaqueta el SavedModel, lo sube a S3 y despliega el endpoint en SageMaker."""
    if not role or not bucket:
        raise ValueError("--role y --bucket son obligatorios para desplegar.")

    archive = package_saved_model(SAVED_MODEL_DIR, PACKAGE_PATH)
    session = sagemaker.Session()
    model_data = session.upload_data(
        path=str(archive),
        bucket=bucket,
        key_prefix=f"models/{endpoint_name}",
    )
    print(f"Paquete subido a {model_data}", flush=True)

    tf_model = TensorFlowModel(
        model_data=model_data,
        role=role,
        framework_version="2.13",
        entry_point="inference.py",
        source_dir="sagemaker_code",
        sagemaker_session=session,
    )

    print(f"Desplegando endpoint '{endpoint_name}' ({instance_type})...", flush=True)
    tf_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )

    class_names = read_json(str(CLASSES_PATH))["class_names"]
    payload = {
        "endpoint_name": endpoint_name,
        "model_data": model_data,
        "CLASS_NAMES": ",".join(class_names),
    }
    write_json(DEPLOY_REPORT_PATH, payload)
    print(f"Despliegue exitoso. Reporte guardado en {DEPLOY_REPORT_PATH}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role")
    ap.add_argument("--bucket")
    ap.add_argument("--endpoint-name", default="industrial-vgg16")
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--delete", action="store_true")
    args = ap.parse_args()

    if args.delete:
        delete_endpoint(args.endpoint_name)
    else:
        deploy_endpoint(args.role, args.bucket, args.endpoint_name, args.instance_type)


if __name__ == "__main__":
    main()
