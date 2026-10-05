"""Paso 3 · Validación: una sola partición frente a la validación cruzada, y qué créditos malos detecta el modelo."""
import numpy as np                                                 # numpy: para trabajar con listas de números
from sklearn.base import clone                                     # clone: crea una copia limpia (sin entrenar) de un modelo
from sklearn.metrics import recall_score                           # recall: de los positivos reales, cuántos encontró el modelo
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split


def validar(pipeline, X, y, cv):
    """Validación cruzada: devuelve la media y la desviación de la exactitud."""
    puntajes = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")  # una exactitud por pliegue
    return puntajes.mean(), puntajes.std()


def proporcion_malos_detectados(pipeline, X, y, cv):
    """De todos los créditos malos, ¿qué proporción marca el modelo como malos?"""
    pred = cross_val_predict(pipeline, X, y, cv=cv)                 # cada caso predicho por un modelo que no lo vio
    return recall_score(y, pred, pos_label=1)                       # positivo = 1 = crédito malo


def particion_unica(pipeline, X, y, semillas=range(10)):
    """Repite una partición 75/25 con varias semillas y devuelve la exactitud de cada una."""
    accs = []
    for s in semillas:
        X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.25, stratify=y, random_state=s)
        modelo = clone(pipeline).fit(X_ent, y_ent)                  # copia limpia entrenada solo con el 75%
        accs.append(modelo.score(X_pru, y_pru))                     # exactitud en el 25% restante
    return np.array(accs)
