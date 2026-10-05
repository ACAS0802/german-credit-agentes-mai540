"""Métricas de la tarea (rol ingeniero de modelos): árbol y Random Forest.

Ejecutar desde la raíz del proyecto:  python experimentos.py
Escribe entregas/02_ingeniero_metricas.json.

Métricas principales: usan el `cv` de datos.py sin cambios.
Secciones de sensibilidad (respuesta a la auditoría H1, H2, H3, H5): son descriptivas y no
sustituyen al esquema fijo.
"""
import json
import os

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split

from datos import cargar_datos
from modelos import crear_pipeline, tres_modelos
from validacion import particion_unica, proporcion_malos_detectados, validar

SEMILLAS = list(range(10))          # particiones 75/25 con semillas 0..9
SEMILLAS_CV = list(range(20))       # H3: semillas alternativas de StratifiedKFold (solo sensibilidad)
UMBRALES = [0.3, 0.4, 0.5]          # H2: umbrales sobre probabilidades de CV (solo descriptivo)
MODELOS = ("Árbol", "Random Forest")


def r(x):
    return round(float(x), 4)


def resumen(valores, cv_media=None):
    v = np.asarray(valores, dtype=float)
    d = {
        "valores": [r(a) for a in v],
        "minimo": r(v.min()),
        "maximo": r(v.max()),
        "rango": r(v.max() - v.min()),
        "media": r(v.mean()),
        "desviacion_ddof0": r(v.std(ddof=0)),
        "desviacion_ddof1": r(v.std(ddof=1)),
    }
    if cv_media is not None:
        d["media_menos_cv"] = r(v.mean() - cv_media)
        d["minimo_menos_cv"] = r(v.min() - cv_media)
        d["maximo_menos_cv"] = r(v.max() - cv_media)
    return d


def particion_no_estratificada(pipe, X, y, semillas):
    accs = []
    for s in semillas:
        X_e, X_v, y_e, y_v = train_test_split(X, y, test_size=0.25, random_state=s)
        accs.append(clone(pipe).fit(X_e, y_e).score(X_v, y_v))
    return np.array(accs)


def main():
    X, y, cols_cat, cv = cargar_datos()
    resultados = {}
    for nombre in MODELOS:
        pipe = crear_pipeline(tres_modelos()[nombre], cols_cat)

        # --- Métricas principales (cv fijo de datos.py) ---
        media, desv = validar(pipe, X, y, cv)                       # desv. con ddof=0 (numpy.std)
        pliegues = cross_val_score(clone(pipe), X, y, cv=cv)         # mismas 5 exactitudes, para H1
        malos = proporcion_malos_detectados(pipe, X, y, cv)          # umbral por defecto (predict)
        pred = cross_val_predict(clone(pipe), X, y, cv=cv)
        accs = particion_unica(pipe, X, y, semillas=SEMILLAS)        # estratificada

        # --- H2: umbrales (descriptivo; NO se usa para elegir umbral) ---
        proba = cross_val_predict(clone(pipe), X, y, cv=cv, method="predict_proba")[:, 1]
        umbrales = {}
        for u in UMBRALES:
            p = (proba >= u).astype(int)
            umbrales[f"proba>={u}"] = {
                "recall_bad": r(recall_score(y, p)),
                "exactitud": r(accuracy_score(y, p)),
                "proporcion_predicha_bad": r(p.mean()),
            }

        # --- H3: sensibilidad de la media de CV a la semilla del barajado ---
        medias_cv = [cross_val_score(clone(pipe), X, y,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean()
                     for s in SEMILLAS_CV]
        mcv = np.array(medias_cv)

        # --- H5: partición única sin estratificar ---
        accs_ne = particion_no_estratificada(pipe, X, y, SEMILLAS)

        resultados[nombre] = {
            "modelo": repr(tres_modelos()[nombre]),
            "cv_exactitud_media": r(media),
            "cv_exactitud_desviacion": r(desv),            # se conserva por compatibilidad: ddof=0
            "cv_exactitud_desviacion_ddof": 0,
            "cv_pliegues": resumen(pliegues),
            "recall_bad_cross_val_predict": r(malos),
            "recall_bad_umbral": "predict() del modelo (argmax de probabilidades; empate 0.5 → good)",
            "proporcion_predicha_bad": r(pred.mean()),
            "particion_unica": dict(semillas=SEMILLAS, estratificada=True, **resumen(accs, media)),
            "comparacion_homogenea": {
                "rango_particion_unica": r(accs.max() - accs.min()),
                "rango_pliegues_cv": r(pliegues.max() - pliegues.min()),
                "desviacion_ddof1_particion_unica": r(accs.std(ddof=1)),
                "desviacion_ddof1_pliegues_cv": r(pliegues.std(ddof=1)),
            },
            "sensibilidad_umbral_descriptiva": umbrales,
            "sensibilidad_semilla_cv": {
                "semillas": SEMILLAS_CV,
                "medias": [r(a) for a in mcv],
                "minimo": r(mcv.min()),
                "maximo": r(mcv.max()),
                "promedio": r(mcv.mean()),
                "n_semillas_media_mayor_0.70": int((mcv > 0.70).sum()),
                "media_particion_unica_menos_promedio_cv": r(accs.mean() - mcv.mean()),
            },
            "particion_unica_no_estratificada": dict(semillas=SEMILLAS, estratificada=False,
                                                     **resumen(accs_ne, media)),
        }
        print(f"{nombre:14s} CV {media:.3f} ± {pliegues.std(ddof=1):.3f} (ddof=1) | recall bad {malos:.3f} "
              f"| predice bad {pred.mean():.3f} | partición única {accs.min():.3f}–{accs.max():.3f} "
              f"| CV semillas 0–19 {mcv.min():.3f}–{mcv.max():.3f}")

    salida = {
        "configuracion": {
            "validacion_cruzada": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) de datos.py",
            "particion_unica": "train_test_split(test_size=0.25, stratify=y, random_state=s), s=0..9",
            "preprocesamiento": "ColumnTransformer: OrdinalEncoder(handle_unknown='use_encoded_value', "
                                 "unknown_value=-1) en 13 categóricas; 7 numéricas passthrough",
            "random_state_modelos": 42,
            "clase_positiva": "bad (y=1)",
            "referencia_siempre_bueno": r(1 - y.mean()),
            "desviaciones": "ddof=0 = numpy.std poblacional (validar()); ddof=1 = muestral",
            "notas_sensibilidad": "sensibilidad_umbral y sensibilidad_semilla_cv son descriptivas; "
                                  "no se usaron para elegir umbral ni esquema",
            "sklearn_version": sklearn.__version__,
        },
        "modelos": resultados,
    }
    os.makedirs("entregas", exist_ok=True)
    with open("entregas/02_ingeniero_metricas.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)
    print("Escrito entregas/02_ingeniero_metricas.json")


if __name__ == "__main__":
    main()
