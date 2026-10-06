"""Paso 6 · Auditoría: ¿el modelo se equivoca más con algún subgrupo?"""
import pandas as pd
from sklearn.model_selection import cross_val_predict


def exactitud_por_subgrupo(pipeline, X, y, cv, columna):
    """Exactitud y número de casos de cada grupo de una columna (p. ej. 'housing' o 'personal_status')."""
    pred = cross_val_predict(pipeline, X, y, cv=cv)   # cada caso lo predice un modelo que no lo vio
    tabla = pd.DataFrame({"grupo": X[columna].values, "real": y.values, "pred": pred})
    tabla["acierto"] = tabla["real"] == tabla["pred"]
    malos = tabla[tabla["real"] == 1]
    return pd.DataFrame({
        "casos": tabla.groupby("grupo").size(),
        "exactitud": tabla.groupby("grupo")["acierto"].mean(),
        "prop_malos": tabla.groupby("grupo")["real"].mean(),
        "malos_detectados": malos.groupby("grupo")["acierto"].mean(),   # recall de «bad» dentro del grupo
    })
