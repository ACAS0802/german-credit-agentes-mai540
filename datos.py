"""Paso 1 · Preparar los datos: leer el CSV y dejar listo X, y, las columnas categóricas y la validación cruzada."""
import pandas as pd                                    # pandas: la librería para trabajar con tablas (como Excel, pero en código)
from sklearn.model_selection import StratifiedKFold    # StratifiedKFold: reparte los datos en pliegues que conservan la proporción de clases


def cargar_datos(ruta="datos/credit_g.csv"):
    """Devuelve X (variables), y (objetivo), cols_cat (columnas con texto) y cv (el esquema de validación)."""
    df = pd.read_csv(ruta)                                            # 1,000 filas: 20 variables + la columna 'clase'
    y = (df["clase"] == "bad").astype(int)                            # clase positiva: 1 = crédito malo
    X = df.drop(columns="clase")                                      # todo lo demás son variables
    cols_cat = X.select_dtypes(exclude="number").columns.tolist()    # columnas de texto (se codifican dentro del pipeline)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # 5 pliegues con la misma proporción de malos
    return X, y, cols_cat, cv


if __name__ == "__main__":
    X, y, cols_cat, cv = cargar_datos()
    print("Forma de X:", X.shape)                                   # (1000, 20): 1,000 solicitudes y 20 variables
    print("Proporción de créditos malos:", round(y.mean(), 2))      # 0.3: el 30% es malo, por eso 'siempre bueno' acierta 70%
    print("Columnas categóricas:", len(cols_cat))                   # 13 columnas con texto
