"""Métricas de la tarea (rol ingeniero de modelos): árbol y Random Forest.

Ejecutar desde la raíz del proyecto:  python experimentos.py
Escribe entregas/02_ingeniero_metricas.json.
"""
import json
import os

import sklearn

from datos import cargar_datos
from modelos import crear_pipeline, tres_modelos
from validacion import particion_unica, proporcion_malos_detectados, validar

SEMILLAS = list(range(10))      # particiones 75/25 con semillas 0..9
MODELOS = ("Árbol", "Random Forest")


def main():
    X, y, cols_cat, cv = cargar_datos()
    resultados = {}
    for nombre in MODELOS:
        pipe = crear_pipeline(tres_modelos()[nombre], cols_cat)
        media, desv = validar(pipe, X, y, cv)
        malos = proporcion_malos_detectados(pipe, X, y, cv)
        accs = particion_unica(pipe, X, y, semillas=SEMILLAS)
        resultados[nombre] = {
            "modelo": repr(tres_modelos()[nombre]),
            "cv_exactitud_media": round(float(media), 4),
            "cv_exactitud_desviacion": round(float(desv), 4),
            "recall_bad_cross_val_predict": round(float(malos), 4),
            "particion_unica": {
                "semillas": SEMILLAS,
                "exactitudes": [round(float(a), 4) for a in accs],
                "minimo": round(float(accs.min()), 4),
                "maximo": round(float(accs.max()), 4),
                "rango": round(float(accs.max() - accs.min()), 4),
                "media": round(float(accs.mean()), 4),
                "media_menos_cv": round(float(accs.mean() - media), 4),
                "minimo_menos_cv": round(float(accs.min() - media), 4),
                "maximo_menos_cv": round(float(accs.max() - media), 4),
            },
        }
        print(f"{nombre:14s} CV {media:.3f} ± {desv:.3f} | recall bad {malos:.3f} | "
              f"partición única {accs.min():.3f}–{accs.max():.3f} (media {accs.mean():.3f})")

    salida = {
        "configuracion": {
            "validacion_cruzada": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) de datos.py",
            "particion_unica": "train_test_split(test_size=0.25, stratify=y, random_state=s), s=0..9",
            "preprocesamiento": "ColumnTransformer: OrdinalEncoder(handle_unknown='use_encoded_value', "
                                 "unknown_value=-1) en 13 categóricas; 7 numéricas passthrough",
            "random_state_modelos": 42,
            "clase_positiva": "bad (y=1)",
            "referencia_siempre_bueno": float(round(1 - y.mean(), 4)),
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
