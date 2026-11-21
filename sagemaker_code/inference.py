"""Handlers de pre/post-procesamiento para el contenedor TensorFlow Serving de SageMaker.

POR COMPLETAR. Consulte la Fase 5 de la guia.

Contrato del endpoint (la Lambda y las pruebas dependen de esto):
  Entrada : bytes de una imagen con Content-Type image/jpeg o image/png
  Salida  : JSON {"predictions": [[p_clase_0, p_clase_1, ...]]}  (una lista de probabilidades por imagen)

El modelo exportado YA incluye el preprocesamiento de VGG16: el handler solo debe
decodificar la imagen, pasarla a RGB, redimensionarla a 224x224 y enviar pixeles 0-255.
"""


def input_handler(data, context):
    """TODO: bytes de imagen -> cuerpo JSON {"instances": [...]} que entiende TF Serving."""
    raise NotImplementedError


def output_handler(response, context):
    """TODO: validar el codigo de estado de TF Serving y devolver (contenido, accept_header)."""
    raise NotImplementedError
