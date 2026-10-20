"""Paso 6 · Auditoría: ¿el modelo se equivoca más con algún subgrupo?"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import cross_val_predict

# Tarea 8.1 · Umbral de disparidad justificado por el costo del proyecto (hallazgo H3 / D3).
#
# Métrica prioritaria: malos detectados (recall de «bad»), porque aprobar un malo cuesta 5.
# Una caída d en los malos detectados de un grupo con m malos son d·m malos aprobados de más,
# es decir 5·d·m unidades de costo (la columna costo_extra lo muestra en cada grupo).
#
# Por qué 0.10 y no otro valor:
#   1. Costo. Por cada solicitante malo, «aprobar a todos» cuesta 5 y el modelo elegido cuesta
#      costo/300 (unos 2.2 con costo ≈ 650), así que el modelo ahorra unas 2.8 unidades por malo.
#      Una caída de 0.10 le quita a ese grupo 5 × 0.10 = 0.5 unidades por malo: cerca de un sexto
#      del beneficio (0.5 / 2.8 ≈ 0.18). Toleramos perder hasta un sexto del beneficio en un grupo;
#      con 0.20 perdería más de un tercio, y eso ya cambiaría la decisión de usarlo con ese grupo.
#   2. Ruido. Con el mínimo de 30 malos por grupo y un recall cercano a 0.7, el error estándar del
#      recall es √(0.7·0.3/30) ≈ 0.08. Un umbral menor que 0.10 marcaría como disparidad diferencias
#      que el muestreo produce solo; por eso, además, cada caída lleva una prueba de proporciones.
UMBRAL_DISPARIDAD = 0.10
MIN_MALOS = 30                      # por debajo, el grupo se reporta como «no concluyente», nunca como aprobado


def auditar_grupos(y, pred, grupos, costo_malo=5):
    """Desglose por grupo a partir de predicciones fuera de pliegue. Para que sea honesto, cada caso
    debe predecirlo un modelo que no lo vio ni al entrenar ni al elegir sus hiperparámetros (capstone.py
    usa predicciones anidadas). 'grupos' puede ser una columna del modelo o una que solo audita."""
    t = pd.DataFrame({"grupo": np.asarray(grupos), "real": np.asarray(y), "pred": np.asarray(pred)})
    t["acierto"] = t["real"] == t["pred"]
    malos = t[t["real"] == 1]
    recall_global = malos["acierto"].mean()
    filas = {}
    for g, d in t.groupby("grupo"):
        m = d[d["real"] == 1]
        resto = malos[malos["grupo"] != g]
        r = m["acierto"].mean() if len(m) else np.nan
        caida = recall_global - r
        # prueba de dos proporciones: recall del grupo frente al de los demás malos
        p = np.nan
        if len(m) and len(resto):
            pc = (m["acierto"].sum() + resto["acierto"].sum()) / (len(m) + len(resto))
            ee = np.sqrt(pc * (1 - pc) * (1 / len(m) + 1 / len(resto)))
            p = float(2 * norm.sf(abs(r - resto["acierto"].mean()) / ee)) if ee > 0 else 1.0
        concluyente = len(m) >= MIN_MALOS
        veredicto = ("no concluyente" if not concluyente
                     else "supera el umbral" if caida > UMBRAL_DISPARIDAD else "dentro del umbral")
        filas[g] = {"casos": len(d), "exactitud": d["acierto"].mean(), "prop_malos": d["real"].mean(),
                    "malos_en_grupo": len(m), "malos_detectados": r, "caida_vs_global": caida,
                    "costo_extra": max(caida, 0) * len(m) * costo_malo, "p_valor": p,
                    "veredicto": veredicto}
    return pd.DataFrame(filas).T.infer_objects()


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
