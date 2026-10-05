"""Paso 3 · Validación: una sola partición frente a la validación cruzada, y qué créditos malos detecta el modelo."""
import numpy as np                                                 # numpy: para trabajar con listas de números
from sklearn.base import clone                                     # clone: crea una copia limpia (sin entrenar) de un modelo
from sklearn.metrics import recall_score                           # recall: de los positivos reales, cuántos encontró el modelo
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split


def validar(pipeline, X, y, cv):
    """Validación cruzada: devuelve la media y la desviación de la exactitud."""
    # TODO 4 (en clase): cross_val_score → media y desviación
    raise NotImplementedError("TODO 4: completar en clase")


def proporcion_malos_detectados(pipeline, X, y, cv):
    """De todos los créditos malos, ¿qué proporción marca el modelo como malos?"""
    # TODO 5 (en clase): cross_val_predict + recall_score
    raise NotImplementedError("TODO 5: completar en clase")


def particion_unica(pipeline, X, y, semillas=range(10)):
    """Repite una partición 75/25 con varias semillas y devuelve la exactitud de cada una."""
    # TODO 6 (en clase): bucle con train_test_split y varias semillas
    raise NotImplementedError("TODO 6: completar en clase")
