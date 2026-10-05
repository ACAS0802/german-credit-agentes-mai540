"""Paso 1 · Preparar los datos: leer el CSV y dejar listo X, y, las columnas categóricas y la validación cruzada."""
import pandas as pd                                    # pandas: la librería para trabajar con tablas (como Excel, pero en código)
from sklearn.model_selection import StratifiedKFold    # StratifiedKFold: reparte los datos en pliegues que conservan la proporción de clases


def cargar_datos(ruta="datos/credit_g.csv"):
    """Devuelve X (variables), y (objetivo), cols_cat (columnas con texto) y cv (el esquema de validación)."""
    # TODO 1 (en clase): leer el CSV, separar y/X, detectar columnas categóricas y crear cv
    raise NotImplementedError("TODO 1: completar en clase")


if __name__ == "__main__":
    X, y, cols_cat, cv = cargar_datos()
    print("Forma de X:", X.shape)                                   # (1000, 20): 1,000 solicitudes y 20 variables
    print("Proporción de créditos malos:", round(y.mean(), 2))      # 0.3: el 30% es malo, por eso 'siempre bueno' acierta 70%
    print("Columnas categóricas:", len(cols_cat))                   # 13 columnas con texto
