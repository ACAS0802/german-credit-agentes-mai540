"""Tarea 8.1 · Flujo final del capstone: capacidad, número honesto, selección anidada, robustez,
auditoría por subgrupo y errores por categoría del modelo que elige la regla.

Ejecutar desde la raíz:  python capstone.py      (unos 25 minutos con 2 núcleos)
                  python capstone.py --auditoria   (solo el paso 6, un par de minutos)
Escribe resultados/capstone.json, resultados/capacidad_boosting.csv, resultados/subgrupos/*.csv,
resultados/errores/*.csv y figs/capacidad_boosting.png.

Protocolo común (el mismo para todos los modelos, igual que en la 6.2):
- Validación cruzada externa: StratifiedKFold(5, shuffle=True, random_state=42) de datos.py.
- Las cifras que se reportan (pasos 2, 3, 4 y 6) son anidadas: toda búsqueda o elección ocurre
  dentro de una validación interna (random_state=1) que solo ve el entrenamiento de cada pliegue
  externo, así que ningún caso de prueba influye en los hiperparámetros con que se le predice.
- Dos pasos usan los 1,000 casos a propósito y no producen cifras de desempeño: el 1 (explorar la
  capacidad, para ver dónde memoriza) y el 5 (elegir y entrenar el modelo final de la herramienta).
- Ningún modelo comparado usa VARIABLES_SENSIBLES; se usan solo para auditar.
- Regla de elección: regla.py (exactitud > 0.70 y, entre esos, menor costo 5:1).
"""
import json
import sys
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from auditoria import MIN_MALOS, UMBRAL_DISPARIDAD, auditar_grupos
from busqueda import PARAM_GRID
from datos import cargar_datos
from modelos import PARAM_GRID_BOOSTING, VARIABLES_SENSIBLES, boosting_afinado, crear_pipeline, modelo_elegido
from regla import EXACTITUD_MINIMA, indice_por_regla
from seleccion import SeleccionPorRegla, evaluar_por_pliegues

CV_INTERNA = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)   # distinta de la externa
N_JOBS = -1                                                               # usa todos los núcleos


def candidatos(cols_cat):
    """Los siete candidatos que compiten. Los que tienen búsqueda la hacen dentro de CV_INTERNA."""
    def p(modelo):
        return crear_pipeline(modelo, cols_cat, VARIABLES_SENSIBLES)
    return {
        "Árbol afinado": GridSearchCV(p(DecisionTreeClassifier(random_state=42)), PARAM_GRID,
                                      cv=CV_INTERNA, scoring="accuracy", n_jobs=N_JOBS),
        "Árbol afinado balanceado": GridSearchCV(p(DecisionTreeClassifier(random_state=42, class_weight="balanced")),
                                                 PARAM_GRID, cv=CV_INTERNA, scoring="accuracy", n_jobs=N_JOBS),
        "Random Forest": p(RandomForestClassifier(random_state=42)),
        "Random Forest balanceado": p(RandomForestClassifier(random_state=42, class_weight="balanced")),
        "Boosting": p(HistGradientBoostingClassifier(random_state=42)),
        "Boosting balanceado (6.2)": p(HistGradientBoostingClassifier(class_weight="balanced", random_state=42)),
        "Boosting balanceado afinado (8.1)": boosting_afinado(cols_cat, CV_INTERNA, n_jobs=N_JOBS),
    }


def capacidad(X, y, cols_cat, cv):
    """E3: exploración de capacidad con entrenamiento y validación juntos. Usa CV_INTERNA, la misma
    partición con la que la elección final (paso 5) afina el boosting, para que la tabla y la figura
    muestren exactamente la búsqueda que produjo el modelo elegido. (Con la partición externa gana
    otra combinación, 4 hojas: varias quedan a menos de 10 unidades de costo entre sí.)"""
    gs = boosting_afinado(cols_cat, CV_INTERNA, n_jobs=N_JOBS).fit(X, y)
    r = pd.DataFrame(gs.cv_results_)
    tabla = pd.DataFrame({
        "max_leaf_nodes": r["param_modelo__max_leaf_nodes"], "min_samples_leaf": r["param_modelo__min_samples_leaf"],
        "learning_rate": r["param_modelo__learning_rate"], "max_iter": r["param_modelo__max_iter"],
        "exactitud_entrenamiento": r["mean_train_exactitud"].round(4),
        "exactitud_validacion": r["mean_test_exactitud"].round(4),
        "brecha": (r["mean_train_exactitud"] - r["mean_test_exactitud"]).round(4),
        "costo_validacion_1000_casos": (-r["mean_test_costo"] * CV_INTERNA.get_n_splits()).round(0).astype(int),
    })
    tabla["cumple_exactitud"] = tabla["exactitud_validacion"] > EXACTITUD_MINIMA
    tabla.sort_values("costo_validacion_1000_casos").to_csv("resultados/capacidad_boosting.csv", index=False)

    # Figura: a igualdad de lo demás (lo del ganador), cómo cambian entrenamiento, validación y costo con las hojas
    g = {k.replace("modelo__", ""): v for k, v in gs.best_params_.items()}
    sub = tabla[(tabla.min_samples_leaf == g["min_samples_leaf"]) & (tabla.learning_rate == g["learning_rate"])
                & (tabla.max_iter == g["max_iter"])].sort_values("max_leaf_nodes")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(sub.max_leaf_nodes, sub.exactitud_entrenamiento, "o-", color="#2a78d6", lw=2, label="entrenamiento")
    a1.plot(sub.max_leaf_nodes, sub.exactitud_validacion, "s-", color="#eb6834", lw=2, label="validación")
    a1.axhline(EXACTITUD_MINIMA, ls="--", color="#888", lw=1)
    a1.set_xscale("log", base=2); a1.set_xticks(sub.max_leaf_nodes); a1.set_xticklabels(sub.max_leaf_nodes)
    a1.set_xlabel("hojas por árbol (max_leaf_nodes)"); a1.set_ylabel("exactitud"); a1.legend(fontsize=8)
    a1.set_title("Más hojas: el entrenamiento sube, la validación no", fontsize=10, loc="left")
    a2.plot(sub.max_leaf_nodes, sub.costo_validacion_1000_casos, "o-", color="#444", lw=2)
    a2.set_xscale("log", base=2); a2.set_xticks(sub.max_leaf_nodes); a2.set_xticklabels(sub.max_leaf_nodes)
    a2.set_xlabel("hojas por árbol (max_leaf_nodes)"); a2.set_ylabel("costo (5 × malo aprobado + bueno rechazado)")
    a2.set_title("Costo en validación", fontsize=10, loc="left")
    for a in (a1, a2):
        a.grid(alpha=.3)
    fig.suptitle(f"Boosting balanceado · fijos: tasa {g['learning_rate']}, {g['max_iter']} árboles, "
                 f"{g['min_samples_leaf']} casos por hoja (los del modelo elegido)", fontsize=9)
    fig.tight_layout(); fig.savefig("figs/capacidad_boosting.png", dpi=200); plt.close(fig)
    return {"ganador_en_toda_la_muestra": g,
            "nota": "Exploración para ver la capacidad (E3). Su puntaje es optimista; el número honesto es la anidada."}


def robustez(X, y, cands, semillas=range(5)):
    """H4: ¿la regla elige lo mismo con otras particiones? Se evalúan TODOS los candidatos."""
    salida = {}
    for s in semillas:
        cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        fila = {n: {k: evaluar_por_pliegues(e, X, y, cv_s)[0][k] for k in ("exactitud_media", "malos_detectados", "costo")}
                for n, e in cands.items()}
        nombres = list(fila)
        fila["gana_por_regla"] = nombres[indice_por_regla([fila[n]["exactitud_media"] for n in nombres],
                                                          [fila[n]["costo"] for n in nombres])]
        salida[s] = fila
        print(f"  semilla {s}: gana {fila['gana_por_regla']}")
    return salida


def por_grupo(y, pred, grupos, carpeta, nombre):
    tabla = auditar_grupos(y, pred, grupos)
    buenos = pd.DataFrame({"g": np.asarray(grupos), "r": np.asarray(y), "p": pred})
    buenos = buenos[buenos.r == 0]
    tabla["buenos_rechazados"] = buenos.groupby("g")["p"].mean()       # proporción de buenos que el modelo rechaza
    tabla.round(4).to_csv(f"resultados/{carpeta}/{nombre}.csv")
    return tabla.round(4).to_dict(orient="index")


def auditar_elegido(s, X, y, cols_cat, cv):
    """Paso 6: auditoría por subgrupo y errores por categoría. Tarda un par de minutos y se puede
    repetir sola con  python capstone.py --auditoria  (reutiliza los pasos 1 a 5 ya guardados).

    Las predicciones salen del procedimiento que produjo el modelo, anidado: en cada pliegue externo
    el boosting se afina solo con el entrenamiento (hallazgo H1 de la tercera auditoría). Si se usara
    modelo_elegido() con sus hiperparámetros ya fijos, cada caso de prueba habría influido en elegirlos.
    Esa versión se guarda solo como referencia («modelo_elegido_misma_cv», optimista)."""
    s["modelo_elegido_misma_cv"], _ = evaluar_por_pliegues(
        crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES), X, y, cv)
    s["procedimiento_anidado"], pred = evaluar_por_pliegues(
        boosting_afinado(cols_cat, CV_INTERNA, n_jobs=N_JOBS), X, y, cv)
    edad = pd.cut(X["age"], [0, 24, 34, 49, 120], labels=["19-24", "25-34", "35-49", "50+"]).astype(str)
    sexo = X["personal_status"].str.split().str[0]                                  # male / female
    s["subgrupos"] = {n: por_grupo(y, pred, g, "subgrupos", n) for n, g in
                      (("edad_grupo", edad), ("sexo", sexo), ("personal_status", X["personal_status"]),
                       ("foreign_worker", X["foreign_worker"]), ("housing", X["housing"]))}
    duracion = pd.cut(X["duration"], [0, 12, 24, 36, 100], labels=["≤12 meses", "13-24", "25-36", ">36"]).astype(str)
    monto = pd.qcut(X["credit_amount"], 4, labels=["Q1 bajo", "Q2", "Q3", "Q4 alto"]).astype(str)
    s["errores_por_categoria"] = {n: por_grupo(y, pred, g, "errores", n) for n, g in
                                  (("proposito", X["purpose"]), ("duracion", duracion), ("monto", monto),
                                   ("cuenta_corriente", X["checking_status"]))}


def main():
    t0 = time.time()
    for d in ("resultados/subgrupos", "resultados/errores", "figs"):
        os.makedirs(d, exist_ok=True)
    X, y, cols_cat, cv = cargar_datos()
    if "--auditoria" in sys.argv:                       # solo el paso 6, sobre lo ya calculado
        with open("resultados/capstone.json", encoding="utf-8") as f:
            s = json.load(f)
        print("6. Auditoría por subgrupo y errores por categoría (predicciones anidadas)")
        auditar_elegido(s, X, y, cols_cat, cv)
        with open("resultados/capstone.json", "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2, default=str)
        print("   anidado:", s["procedimiento_anidado"])
        return
    cands = candidatos(cols_cat)
    s = {"configuracion": {
        "cv_externa": "StratifiedKFold(5, shuffle=True, random_state=42) de datos.py",
        "cv_interna": "StratifiedKFold(5, shuffle=True, random_state=1)",
        "regla": f"exactitud > {EXACTITUD_MINIMA} y, entre esos, menor costo (regla.py)",
        "espacio_capacidad_boosting": PARAM_GRID_BOOSTING,
        "umbral_disparidad": f"{UMBRAL_DISPARIDAD} en malos detectados; concluyente si >= {MIN_MALOS} malos",
        "variables_excluidas_del_modelo": list(VARIABLES_SENSIBLES),
        "sklearn": sklearn.__version__}}

    print("1. Capacidad del boosting (E3)")
    s["capacidad"] = capacidad(X, y, cols_cat, cv)

    print("2. Comparación de los siete candidatos con el mismo protocolo")
    s["comparacion"] = {}
    for n, e in cands.items():
        s["comparacion"][n] = evaluar_por_pliegues(e, X, y, cv)[0]
        r = s["comparacion"][n]
        print(f"  {n:36s} {r['exactitud_media']:.3f}±{r['exactitud_desv_ddof1']:.3f}  malos {r['malos_detectados']:.3f}"
              f"  costo {r['costo']:4d}  brecha {r['brecha']:.3f}")

    print("3. Selección anidada: la regla se aplica dentro de cada pliegue externo (A6)")
    s["seleccion_anidada"], _ = evaluar_por_pliegues(SeleccionPorRegla(cands, CV_INTERNA), X, y, cv)
    print("  ", s["seleccion_anidada"])

    print("4. Robustez con las semillas 0 a 4 (todos los candidatos)")
    s["robustez_semillas_cv"] = robustez(X, y, cands)

    print("5. Elección final con todos los datos")
    final = SeleccionPorRegla(cands, cv).fit(X, y)
    params = (final.modelo_.best_params_ if hasattr(final.modelo_, "best_params_") else {})
    s["eleccion_final"] = {"modelo": final.elegido_,
                           "hiperparametros": {k.replace("modelo__", ""): v for k, v in params.items()},
                           "tabla_de_la_regla": final.tabla_}
    print("  ", s["eleccion_final"]["modelo"], s["eleccion_final"]["hiperparametros"])

    print("6. Auditoría por subgrupo y errores por categoría (predicciones anidadas)")
    auditar_elegido(s, X, y, cols_cat, cv)

    with open("resultados/capstone.json", "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2, default=str)
    print(f"tiempo total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
