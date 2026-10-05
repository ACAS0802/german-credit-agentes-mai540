"""Paso 2 · Los modelos y el Pipeline: la codificación va DENTRO del pipeline para que nada se filtre entre pliegues."""
from sklearn.compose import ColumnTransformer          # ColumnTransformer: aplica una transformación solo a ciertas columnas
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # los dos modelos de conjunto
from sklearn.pipeline import Pipeline                  # Pipeline: encadena pasos (preparar → modelo) como una sola pieza
from sklearn.preprocessing import OrdinalEncoder       # OrdinalEncoder: convierte cada categoría de texto en un número
from sklearn.tree import DecisionTreeClassifier        # el árbol de decisión: nuestro modelo individual


def crear_pipeline(modelo, cols_cat):
    """Envuelve un modelo con su preparación de datos."""
    # TODO 2 (en clase): OrdinalEncoder + ColumnTransformer + Pipeline
    raise NotImplementedError("TODO 2: completar en clase")


def tres_modelos():
    """Los tres modelos que comparamos, con semilla fija."""
    # TODO 3 (en clase): diccionario con Árbol, Random Forest y Boosting
    raise NotImplementedError("TODO 3: completar en clase")
