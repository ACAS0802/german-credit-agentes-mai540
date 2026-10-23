"""Tarea 8.2 · Simulación de degradación con el modelo de la herramienta.

Adaptada de ejemplo_degradacion/simular_degradacion.py del Kit del Módulo 8. Allí se mide solo la
exactitud de un Random Forest; aquí se usa modelo_elegido() (sin variables sensibles) y se mide lo que
decide este proyecto: malos detectados y costo por solicitud (5 × malo aprobado + 1 × bueno rechazado).

Dos cambios de «mundo», cada uno con 20 repartos distintos:
  1. Duración: se entrena con créditos de hasta 24 meses y se usa con créditos más largos
     (el mismo experimento del ejemplo del profesor).
  2. Cuenta corriente: se entrena con solicitantes que tienen cuenta corriente y se usa con quienes
     no tienen (la categoría donde el modelo ya falla más, según resultados/errores/).

Uso:  python degradacion.py      Escribe resultados/degradacion.json

Lección principal: en el cambio 2 la exactitud SUBE (0.66 → 0.70) mientras los malos detectados CAEN
(0.62 → 0.42). Quien vigile solo la exactitud no vería que el modelo empeoró en lo que importa.
"""
import json
import warnings

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from datos import cargar_datos
from modelos import VARIABLES_SENSIBLES, crear_pipeline, modelo_elegido

warnings.filterwarnings("ignore")


def medir(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"exactitud": (tn + tp) / len(y), "malos_detectados": tp / (tp + fn) if tp + fn else np.nan,
            "costo_por_solicitud": (5 * fn + fp) / len(y)}


def simular(X, y, cols_cat, antes, nombre, repartos=20):
    despues = ~antes
    mismo, cambio = [], []
    for semilla in range(repartos):
        X_ent, X_val, y_ent, y_val = train_test_split(X[antes], y[antes], test_size=0.25,
                                                      stratify=y[antes], random_state=semilla)
        m = crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES).fit(X_ent, y_ent)
        mismo.append(medir(y_val, m.predict(X_val)))
        cambio.append(medir(y[despues], m.predict(X[despues])))
    res = {"cambio": nombre, "casos_antes": int(antes.sum()), "casos_despues": int(despues.sum()),
           "prop_malos_antes": round(float(y[antes].mean()), 3), "prop_malos_despues": round(float(y[despues].mean()), 3),
           # Referencia: «aprobar a todos» cuesta 5 por cada malo, es decir 5 × proporción de malos por solicitud
           "costo_aprobar_a_todos_antes": round(5 * float(y[antes].mean()), 2),
           "costo_aprobar_a_todos_despues": round(5 * float(y[despues].mean()), 2)}
    for clave, filas in (("mismo_mundo", mismo), ("mundo_cambiado", cambio)):
        res[clave] = {k: {"media": round(float(np.mean([f[k] for f in filas])), 3),
                          "desv": round(float(np.std([f[k] for f in filas], ddof=1)), 3)} for k in filas[0]}
    return res


def main():
    X, y, cols_cat, _ = cargar_datos()
    salida = [simular(X, y, cols_cat, X["duration"] <= 24, "duración: entrenado con ≤ 24 meses, usado con > 24"),
              simular(X, y, cols_cat, X["checking_status"] != "no checking",
                      "cuenta corriente: entrenado con quienes tienen cuenta, usado con quienes no")]
    with open("resultados/degradacion.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)
    for r in salida:
        print(f"\n{r['cambio']}  (malos: antes {r['prop_malos_antes']:.2f} | después {r['prop_malos_despues']:.2f})")
        for k in ("exactitud", "malos_detectados", "costo_por_solicitud"):
            a, b = r["mismo_mundo"][k], r["mundo_cambiado"][k]
            print(f"  {k:20s} mismo mundo {a['media']:.2f} ± {a['desv']:.2f}   →   mundo cambiado {b['media']:.2f} ± {b['desv']:.2f}")
        print(f"  {'aprobar a todos':20s} costo {r['costo_aprobar_a_todos_antes']:.2f}   →   {r['costo_aprobar_a_todos_despues']:.2f}")


if __name__ == "__main__":
    main()
