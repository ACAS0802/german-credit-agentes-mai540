"""Paso 3 · Validación: una sola partición frente a la validación cruzada, y qué créditos malos detecta el modelo."""
import numpy as np                                                 # numpy: para trabajar con listas de números
from sklearn.base import clone                                     # clone: crea una copia limpia (sin entrenar) de un modelo
from sklearn.metrics import recall_score                           # recall: de los positivos reales, cuántos encontró el modelo
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split


def validar(pipeline, X, y, cv):
    """Validación cruzada: devuelve la media y la desviación de la exactitud."""
    puntajes = cross_val_score(clone(pipeline), X, y, cv=cv, scoring="accuracy")  # una exactitud por pliegue
    return puntajes.mean(), puntajes.std()


def proporcion_malos_detectados(pipeline, X, y, cv):
    """De todos los créditos malos, ¿qué proporción marca el modelo como malos?"""
    pred = cross_val_predict(clone(pipeline), X, y, cv=cv)    # cada predicción la hace un modelo que no vio esa fila
    return recall_score(y, pred, pos_label=1)                 # 1 = crédito malo


def particion_unica(pipeline, X, y, semillas=range(10)):
    """Repite una partición 75/25 con varias semillas y devuelve la exactitud de cada una."""
    exactitudes = []
    for semilla in semillas:
        X_ent, X_val, y_ent, y_val = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=semilla)   # 750 / 250, misma proporción de malos
        modelo = clone(pipeline).fit(X_ent, y_ent)                    # copia limpia en cada repetición
        exactitudes.append(modelo.score(X_val, y_val))
    return np.array(exactitudes)
