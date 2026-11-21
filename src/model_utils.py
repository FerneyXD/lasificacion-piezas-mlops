"""Utilidades con TensorFlow: capa de preprocesamiento VGG16 y carga de datasets."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="industrial")
class VGGPreprocess(tf.keras.layers.Layer):
    """Preprocesamiento de VGG16 dentro del modelo (RGB->BGR + resta de media).

    Al ir dentro del grafo, el endpoint solo necesita redimensionar a 224x224
    y enviar pixeles 0-255: se evita el desajuste train/serve.
    """

    def call(self, x):
        return tf.keras.applications.vgg16.preprocess_input(x)

    def get_config(self):
        return super().get_config()


def get_dataset(path, img_size, batch_size, shuffle, class_names=None):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    )


def load_model(path="models/model.keras"):
    return tf.keras.models.load_model(path, custom_objects={"VGGPreprocess": VGGPreprocess})
