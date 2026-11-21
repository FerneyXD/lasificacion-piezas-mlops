"""Despliegue del modelo en un endpoint de inferencia en tiempo real de Amazon SageMaker.

POR COMPLETAR. Consulte la Fase 5 de la guia.

Uso esperado:
  python src/deploy_sagemaker.py --role arn:aws:iam::<cuenta>:role/<rol> --bucket <bucket>
  python src/deploy_sagemaker.py --delete --endpoint-name industrial-vgg16

Salida OBLIGATORIA al desplegar: reports/deploy.json con
  {"endpoint_name": ..., "model_data": ..., "CLASS_NAMES": "clase1,clase2,..."}
(CLASS_NAMES se usa como variable de entorno de la Lambda; el orden debe ser el de models/classes.json).
"""
import argparse

from common import read_json, write_json  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role")
    ap.add_argument("--bucket")
    ap.add_argument("--endpoint-name", default="industrial-vgg16")
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--delete", action="store_true")
    args = ap.parse_args()  # noqa: F841

    # TODO 1 (--delete): eliminar endpoint y su configuracion, y terminar
    # TODO 2: empaquetar models/saved_model/1 en un model.tar.gz con la estructura que exige TF Serving
    # TODO 3: subir el paquete a S3 con el SDK de SageMaker
    # TODO 4: crear el modelo (TensorFlowModel, framework_version 2.13, usando sagemaker_code/inference.py)
    #         y desplegarlo como endpoint en tiempo real
    # TODO 5: escribir reports/deploy.json (ver docstring)
    raise NotImplementedError


if __name__ == "__main__":
    main()
