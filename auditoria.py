"""Paso 6 · Auditoría: ¿el modelo se equivoca más con algún subgrupo?"""
import pandas as pd
from sklearn.model_selection import cross_val_predict


def exactitud_por_subgrupo(pipeline, X, y, cv, columna):
    """Exactitud y número de casos de cada grupo de una columna (p. ej. 'housing' o 'personal_status')."""
    # TODO 10 (en clase): cross_val_predict + groupby por grupo
    raise NotImplementedError("TODO 10: completar en clase")
