"""Paso 2 · Los modelos y el Pipeline: la codificación va DENTRO del pipeline para que nada se filtre entre pliegues."""
from sklearn.compose import ColumnTransformer          # ColumnTransformer: aplica una transformación solo a ciertas columnas
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # los dos modelos de conjunto
from sklearn.pipeline import Pipeline                  # Pipeline: encadena pasos (preparar → modelo) como una sola pieza
from sklearn.preprocessing import OrdinalEncoder       # OrdinalEncoder: convierte cada categoría de texto en un número
from sklearn.tree import DecisionTreeClassifier        # el árbol de decisión: nuestro modelo individual


def crear_pipeline(modelo, cols_cat):
    """Envuelve un modelo con su preparación de datos."""
    codificador = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)  # categorías nuevas → -1
    prep = ColumnTransformer([("cat", codificador, cols_cat)], remainder="passthrough")  # numéricas pasan tal cual
    return Pipeline([("prep", prep), ("modelo", modelo)])


def tres_modelos():
    """Los tres modelos que comparamos, con semilla fija."""
    return {
        "Árbol": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Boosting": HistGradientBoostingClassifier(random_state=42),
    }
