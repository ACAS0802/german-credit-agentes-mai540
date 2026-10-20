"""Tarea 8.1 · La regla de elección escrita como código (hallazgos A6 y E4 de la auditoría).

En la Tarea 6.2 la regla estaba escrita solo en un comentario y el modelo elegido se fijó a mano.
Aquí la regla es una función: cualquier búsqueda o selección del proyecto la aplica igual.

Regla: entre los modelos cuya exactitud media supera 0.70 (lo que acierta «siempre bueno»),
gana el de menor costo. Costo = 5 × (malos aprobados) + 1 × (buenos rechazados), la matriz de
costos del German Credit original (Hofmann, 1994): aprobar un crédito malo cuesta 5 veces más
que rechazar uno bueno.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer

COSTO_MALO_APROBADO, COSTO_BUENO_RECHAZADO = 5, 1
EXACTITUD_MINIMA = 0.70          # «siempre bueno» acierta el 70 %: un modelo que no lo supera no aporta


def costo(y_real, y_pred):
    """Costo total de los errores. 1 = crédito malo (clase de interés)."""
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    return int(COSTO_MALO_APROBADO * fn + COSTO_BUENO_RECHAZADO * fp)


# Métricas para GridSearchCV. scikit-learn maximiza, por eso el costo entra con signo negativo.
METRICAS = {"exactitud": "accuracy",
            "costo": make_scorer(costo, greater_is_better=False)}


def indice_por_regla(exactitudes, costos):
    """Devuelve la posición del ganador según la regla. Si nadie supera 0.70, el de mayor exactitud."""
    exactitudes, costos = np.asarray(exactitudes, float), np.asarray(costos, float)
    elegibles = np.flatnonzero(exactitudes > EXACTITUD_MINIMA)
    if elegibles.size == 0:
        return int(np.argmax(exactitudes))
    return int(elegibles[np.argmin(costos[elegibles])])


def refit_por_regla(cv_results):
    """Para GridSearchCV(refit=...): elige la combinación con la misma regla."""
    return indice_por_regla(cv_results["mean_test_exactitud"], -np.asarray(cv_results["mean_test_costo"]))
