"""Tarea 8.1 · Selección anidada con la regla del proyecto (hallazgo H1 / A6 de la auditoría).

El problema de la 6.2: se compararon seis modelos con unos pliegues y luego se reportaron las cifras
del ganador medidas con esos mismos pliegues. Eso premia al que tuvo suerte en esa partición.

La solución: «elegir» se vuelve parte del modelo. SeleccionPorRegla recibe los candidatos y, al
entrenarse, compara a todos con una validación cruzada interna que solo ve los datos de entrenamiento,
aplica la regla (regla.py) y reentrena al ganador. Si se evalúa con una validación cruzada externa,
cada pliegue externo elige sin ver sus datos de prueba: el número que sale es el de todo el
procedimiento, no el del ganador con suerte.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate

from regla import COSTO_BUENO_RECHAZADO, COSTO_MALO_APROBADO, METRICAS, indice_por_regla


class SeleccionPorRegla(BaseEstimator, ClassifierMixin):
    """Elige entre 'candidatos' (dict nombre → estimador sin entrenar) con la regla del proyecto."""

    def __init__(self, candidatos, cv_interna):
        self.candidatos = candidatos
        self.cv_interna = cv_interna

    def fit(self, X, y):
        nombres = list(self.candidatos)
        self.tabla_ = {}
        for n in nombres:
            r = cross_validate(clone(self.candidatos[n]), X, y, cv=self.cv_interna, scoring=METRICAS)
            self.tabla_[n] = {"exactitud": float(r["test_exactitud"].mean()),
                              "costo_por_pliegue": float(-r["test_costo"].mean())}
        i = indice_por_regla([self.tabla_[n]["exactitud"] for n in nombres],
                             [self.tabla_[n]["costo_por_pliegue"] for n in nombres])
        self.elegido_ = nombres[i]
        self.modelo_ = clone(self.candidatos[self.elegido_]).fit(X, y)
        self.classes_ = self.modelo_.classes_
        return self

    def predict(self, X):
        return self.modelo_.predict(X)

    def predict_proba(self, X):
        return self.modelo_.predict_proba(X)


def evaluar_por_pliegues(estimador, X, y, cv):
    """Un solo ajuste por pliegue. Devuelve exactitud (media y desviación, ddof=1), exactitud de
    entrenamiento, malos detectados, matriz de confusión acumulada, costo y, si el estimador elige
    algo por dentro (SeleccionPorRegla o GridSearchCV), qué eligió en cada pliegue."""
    acc, acc_ent, eleccion = [], [], []
    pred = np.empty(len(y), dtype=int)
    for ent, pru in cv.split(X, y):
        m = clone(estimador).fit(X.iloc[ent], y.iloc[ent])
        pred[pru] = m.predict(X.iloc[pru])
        acc.append(float((pred[pru] == y.iloc[pru].values).mean()))
        # Exactitud de entrenamiento con predict y no con score(): GridSearchCV.score() falla cuando
        # refit es una función (refit_por_regla), porque busca un puntaje con ese nombre.
        acc_ent.append(float((m.predict(X.iloc[ent]) == y.iloc[ent].values).mean()))
        if hasattr(m, "elegido_"):
            eleccion.append(m.elegido_)
        elif hasattr(m, "best_params_"):
            eleccion.append({k.replace("modelo__", ""): v for k, v in m.best_params_.items()})
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    salida = {
        "exactitud_media": round(float(np.mean(acc)), 4),
        "exactitud_desv_ddof1": round(float(np.std(acc, ddof=1)), 4),
        "exactitud_pliegues": [round(v, 4) for v in acc],
        "exactitud_entrenamiento": round(float(np.mean(acc_ent)), 4),
        "brecha": round(float(np.mean(acc_ent) - np.mean(acc)), 4),
        "malos_detectados": round(float(tp / (tp + fn)), 4),
        "malos_aprobados": int(fn), "buenos_rechazados": int(fp),
        "costo": int(COSTO_MALO_APROBADO * fn + COSTO_BUENO_RECHAZADO * fp),
    }
    if eleccion:
        salida["eleccion_por_pliegue"] = eleccion
    return salida, pred
