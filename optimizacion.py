"""Tarea 6.2 · Búsqueda de hiperparámetros, número honesto, comparación de tres modelos,
curvas de aprendizaje y auditoría por subgrupo del modelo elegido.

Ejecutar desde la raíz:  python optimizacion.py
Escribe resultados/optimizacion.json, resultados/subgrupos_*.csv y figs/*.png.

Protocolo común (el mismo para todos los modelos):
- Validación cruzada externa: la de datos.py, StratifiedKFold(5, shuffle=True, random_state=42).
- Exactitud: media y desviación (ddof=1) de los 5 pliegues.
- Malos detectados: recall de «bad» con cross_val_predict (predicciones fuera del pliegue).
- Costo: 5 × (malos aprobados) + 1 × (buenos rechazados), la matriz de costos del dataset
  original (Hofmann, 1994). Aprobar un crédito malo cuesta 5 veces más.
- Tiempo de ajuste: media por pliegue de cross_validate, con n_jobs=1 para comparar en igualdad.
- Entradas: ningún modelo usa VARIABLES_SENSIBLES (edad, estado civil/sexo, trabajador extranjero);
  se usan solo para auditar (hallazgo E4 de auditoria/AUDIT_REPORT.md).

Regla de elección (hallazgo A6): entre los modelos cuya exactitud media supera 0.70 («siempre
bueno»), el de menor costo; la elección se confirma solo si ese modelo tiene el menor costo también
en las 5 semillas adicionales de la validación cruzada (0 a 4).

Umbral de disparidad (hallazgo D3): caída de 0.10 absoluto en malos detectados de un subgrupo frente
al global (valor por omisión de la Skill). Un subgrupo con menos de 30 malos se marca como no
concluyente (hallazgo D4).
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.tree import DecisionTreeClassifier

from auditoria import exactitud_por_subgrupo
from busqueda import PARAM_GRID, buscar_con_prueba
from curvas import curva_aprendizaje
from datos import cargar_datos
from modelos import VARIABLES_SENSIBLES, crear_pipeline, modelo_elegido, tres_modelos

COSTO_MALO_APROBADO, COSTO_BUENO_RECHAZADO = 5, 1
UMBRAL_DISPARIDAD, MIN_MALOS = 0.10, 30
CV_INTERNA = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)  # distinta de la externa
os.makedirs("resultados", exist_ok=True)
os.makedirs("figs", exist_ok=True)


def evaluar(estimador, X, y, cv):
    """Mismo protocolo para cualquier estimador (pipeline o GridSearchCV)."""
    r = cross_validate(estimador, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    pred = cross_val_predict(estimador, X, y, cv=cv, n_jobs=1)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "exactitud_media": round(float(r["test_score"].mean()), 4),
        "exactitud_desv_ddof1": round(float(r["test_score"].std(ddof=1)), 4),
        "exactitud_pliegues": [round(float(v), 4) for v in r["test_score"]],
        "malos_detectados": round(float(recall_score(y, pred)), 4),
        "malos_aprobados": int(fn), "buenos_rechazados": int(fp),
        "costo": int(COSTO_MALO_APROBADO * fn + COSTO_BUENO_RECHAZADO * fp),
        "tiempo_ajuste_s": round(float(r["fit_time"].mean()), 3),
    }


def main():
    X, y, cols_cat, cv = cargar_datos()
    salida = {"configuracion": {
        "cv_externa": "StratifiedKFold(5, shuffle=True, random_state=42) de datos.py",
        "cv_interna_anidada": "StratifiedKFold(5, shuffle=True, random_state=1)",
        "espacio_busqueda": {k: [str(v) for v in vs] for k, vs in PARAM_GRID.items()},
        "combinaciones": int(np.prod([len(v) for v in PARAM_GRID.values()])),
        "costo": f"{COSTO_MALO_APROBADO} × malo aprobado + {COSTO_BUENO_RECHAZADO} × bueno rechazado",
        "variables_excluidas_del_modelo": list(VARIABLES_SENSIBLES),
        "regla_eleccion": "menor costo entre modelos con exactitud > 0.70, confirmado en semillas 0-4",
        "umbral_disparidad": f"{UMBRAL_DISPARIDAD} en malos detectados; concluyente si >= {MIN_MALOS} malos",
        "sklearn": sklearn.__version__}}

    # 1. Búsqueda sistemática del árbol
    arbol = crear_pipeline(DecisionTreeClassifier(random_state=42), cols_cat, VARIABLES_SENSIBLES)
    gs = GridSearchCV(arbol, PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=1).fit(X, y)
    defecto = evaluar(arbol, X, y, cv)
    afinado_fijo = evaluar(crear_pipeline(DecisionTreeClassifier(
        random_state=42, max_depth=gs.best_params_["modelo__max_depth"],
        min_samples_leaf=gs.best_params_["modelo__min_samples_leaf"]), cols_cat, VARIABLES_SENSIBLES), X, y, cv)
    tabla_gs = pd.DataFrame(gs.cv_results_)[["param_modelo__max_depth", "param_modelo__min_samples_leaf",
                                             "mean_test_score", "std_test_score", "rank_test_score"]]
    tabla_gs.sort_values("rank_test_score").to_csv("resultados/busqueda_arbol.csv", index=False)

    # 2. Número honesto: validación cruzada anidada y prueba reservada
    gs_anidado = GridSearchCV(arbol, PARAM_GRID, cv=CV_INTERNA, scoring="accuracy", n_jobs=1)
    anidada = evaluar(gs_anidado, X, y, cv)
    ganadores = [gs_anidado.fit(X.iloc[ent], y.iloc[ent]).best_params_ for ent, _ in cv.split(X, y)]
    busq_80, prueba_20 = buscar_con_prueba(arbol, X, y, cv)
    salida["busqueda"] = {
        "ganador": {k.replace("modelo__", ""): (None if v is None else int(v)) for k, v in gs.best_params_.items()},
        "puntaje_busqueda_optimista": round(float(gs.best_score_), 4),
        "arbol_por_defecto": defecto,
        "ganador_reevaluado_misma_cv": afinado_fijo,
        "cv_anidada_numero_honesto": anidada,
        "ganadores_por_pliegue_externo": [{k.replace("modelo__", ""): v for k, v in g.items()} for g in ganadores],
        "prueba_reservada_20": {"busqueda_en_80": round(float(busq_80), 4), "prueba": round(float(prueba_20), 4)},
        "optimismo_busqueda": round(float(gs.best_score_ - anidada["exactitud_media"]), 4),
        "mejora_honesta_vs_defecto": round(anidada["exactitud_media"] - defecto["exactitud_media"], 4),
    }

    # 3. Tres modelos con el mismo protocolo (el árbol afinado se afina dentro de cada pliegue)
    modelos = tres_modelos()
    comparacion = {
        "Árbol afinado (anidada)": anidada,
        "Random Forest": evaluar(crear_pipeline(modelos["Random Forest"], cols_cat, VARIABLES_SENSIBLES), X, y, cv),
        "Boosting": evaluar(crear_pipeline(modelos["Boosting"], cols_cat, VARIABLES_SENSIBLES), X, y, cv),
    }
    # Error que más preocupa (aprobar un malo): los mismos modelos con pesos de clase balanceados
    gs_bal = GridSearchCV(crear_pipeline(DecisionTreeClassifier(random_state=42, class_weight="balanced"), cols_cat, VARIABLES_SENSIBLES),
                          PARAM_GRID, cv=CV_INTERNA, scoring="accuracy", n_jobs=1)
    from sklearn.ensemble import RandomForestClassifier
    comparacion["Árbol afinado balanceado (anidada)"] = evaluar(gs_bal, X, y, cv)
    comparacion["Random Forest balanceado"] = evaluar(crear_pipeline(
        RandomForestClassifier(random_state=42, class_weight="balanced"), cols_cat, VARIABLES_SENSIBLES), X, y, cv)
    comparacion["Boosting balanceado (elegido)"] = evaluar(crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES), X, y, cv)
    salida["comparacion"] = comparacion

    # Robustez: ¿la elección depende de la semilla de la validación cruzada?
    robustez = {}
    candidatos = (("Random Forest", lambda: crear_pipeline(modelos["Random Forest"], cols_cat, VARIABLES_SENSIBLES)),
                  ("Boosting", lambda: crear_pipeline(modelos["Boosting"], cols_cat, VARIABLES_SENSIBLES)),
                  ("Árbol afinado balanceado (anidada)", lambda: GridSearchCV(
                      crear_pipeline(DecisionTreeClassifier(random_state=42, class_weight="balanced"), cols_cat, VARIABLES_SENSIBLES),
                      PARAM_GRID, cv=CV_INTERNA, scoring="accuracy", n_jobs=-1)),
                  ("Boosting balanceado", lambda: crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES)))
    for semilla in range(5):
        cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)
        robustez[semilla] = {n: {k: evaluar(f(), X, y, cv_s)[k] for k in ("exactitud_media", "malos_detectados", "costo")}
                             for n, f in candidatos}
    salida["robustez_semillas_cv"] = robustez

    # 4. Curvas de aprendizaje: diagnóstico de la 6.1 (árbol por defecto) frente al elegido
    curvas = {}
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (nombre, pipe) in zip(ejes, (("Árbol por defecto (diagnóstico 6.1)",
                                          crear_pipeline(DecisionTreeClassifier(random_state=42), cols_cat)),  # tal como en la 6.1
                                         ("Boosting balanceado (elegido)", crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES)))):
        t, ent, val = curva_aprendizaje(pipe, X, y, cv)
        curvas[nombre] = {"tamanos": [int(v) for v in t], "entrenamiento": [round(float(v), 4) for v in ent],
                          "validacion": [round(float(v), 4) for v in val],
                          "brecha_final": round(float(ent[-1] - val[-1]), 4)}
        ax.plot(t, ent, "o-", color="#2a78d6", lw=2, label="entrenamiento")
        ax.plot(t, val, "s-", color="#eb6834", lw=2, label="validación")
        ax.axhline(0.70, ls="--", color="#888", lw=1)
        ax.text(t[0], 0.705, "siempre «bueno» 0.70", color="#555", fontsize=8)
        ax.set_title(f"{nombre}\nbrecha final {ent[-1]-val[-1]:.2f}", fontsize=10, loc="left")
        ax.set_xlabel("casos de entrenamiento"); ax.grid(alpha=.3)
    ejes[0].set_ylabel("exactitud"); ejes[0].set_ylim(0.55, 1.02); ejes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig("figs/curvas_6_1_vs_6_2.png", dpi=200); plt.close(fig)
    salida["curvas"] = curvas

    # 5. Auditoría por subgrupo del modelo elegido (las variables sensibles solo auditan)
    elegido = crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES)
    edad = pd.cut(X["age"], [0, 24, 34, 49, 120], labels=["19-24", "25-34", "35-49", "50+"]).astype(str)
    subgrupos = {}
    for col in ("edad_grupo", "personal_status", "housing", "foreign_worker"):
        tabla = (_por_grupo_externo(elegido, X, y, cv, edad) if col == "edad_grupo"
                 else exactitud_por_subgrupo(elegido, X, y, cv, col))
        malos_global = recall_score(y, cross_val_predict(elegido, X, y, cv=cv))
        tabla["malos_en_grupo"] = (tabla["casos"] * tabla["prop_malos"]).round().astype(int)
        tabla["caida_vs_global"] = (malos_global - tabla["malos_detectados"]).round(4)
        tabla["concluyente"] = tabla["malos_en_grupo"] >= MIN_MALOS
        tabla["supera_umbral"] = tabla["concluyente"] & (tabla["caida_vs_global"] > UMBRAL_DISPARIDAD)
        tabla.round(4).to_csv(f"resultados/subgrupos_{col}.csv")
        subgrupos[col] = tabla.round(4).to_dict(orient="index")
    salida["subgrupos_elegido"] = subgrupos

    with open("resultados/optimizacion.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2, default=str)
    _imprimir(salida)


def _por_grupo_externo(pipe, X, y, cv, grupos):
    """Como exactitud_por_subgrupo, para un grupo que no es columna del modelo (la edad agrupada):
    el modelo se entrena con X tal cual; el grupo solo se usa para leer los resultados."""
    pred = cross_val_predict(pipe, X, y, cv=cv)
    t = pd.DataFrame({"grupo": grupos.values, "real": y.values, "pred": pred})
    t["acierto"] = t["real"] == t["pred"]
    m = t[t["real"] == 1]
    return pd.DataFrame({"casos": t.groupby("grupo").size(), "exactitud": t.groupby("grupo")["acierto"].mean(),
                         "prop_malos": t.groupby("grupo")["real"].mean(),
                         "malos_detectados": m.groupby("grupo")["acierto"].mean()})


def _imprimir(s):
    b = s["busqueda"]
    print(f"Búsqueda: ganador {b['ganador']} | puntaje de la búsqueda {b['puntaje_busqueda_optimista']:.3f} | "
          f"anidada (honesto) {b['cv_anidada_numero_honesto']['exactitud_media']:.3f} | "
          f"por defecto {b['arbol_por_defecto']['exactitud_media']:.3f} | prueba reservada {b['prueba_reservada_20']}")
    print(f"{'modelo':38s} exactitud      malos  costo  ajuste")
    for n, r in s["comparacion"].items():
        print(f"{n:38s} {r['exactitud_media']:.3f}±{r['exactitud_desv_ddof1']:.3f}  {r['malos_detectados']:.3f}  "
              f"{r['costo']:5d}  {r['tiempo_ajuste_s']:.3f}s")
    for n, c in s["curvas"].items():
        print(f"curva {n}: entrenamiento {c['entrenamiento'][-1]:.3f} validación {c['validacion'][-1]:.3f} brecha {c['brecha_final']:.3f}")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"tiempo total {time.time()-t0:.0f} s")
