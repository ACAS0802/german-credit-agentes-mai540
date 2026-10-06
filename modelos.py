"""Paso 2 · Los modelos y el Pipeline: la codificación va DENTRO del pipeline para que nada se filtre entre pliegues."""
from sklearn.compose import ColumnTransformer          # ColumnTransformer: aplica una transformación solo a ciertas columnas
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # los dos modelos de conjunto
from sklearn.pipeline import Pipeline                  # Pipeline: encadena pasos (preparar → modelo) como una sola pieza
from sklearn.preprocessing import OrdinalEncoder       # OrdinalEncoder: convierte cada categoría de texto en un número
from sklearn.tree import DecisionTreeClassifier        # el árbol de decisión: nuestro modelo individual


def crear_pipeline(modelo, cols_cat):
    """Envuelve un modelo con su preparación de datos."""
    codificador = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)  # categoría no vista en entrenamiento → -1
    prep = ColumnTransformer(
        [("cat", codificador, cols_cat)],                  # solo las columnas de texto se codifican
        remainder="passthrough",                           # las 7 numéricas pasan tal cual (los árboles no necesitan escalado)
    )
    return Pipeline([("prep", prep), ("modelo", modelo)])  # 'prep' se ajusta solo con los datos de entrenamiento de cada pliegue


def tres_modelos():
    """Los tres modelos que comparamos, con semilla fija."""
    return {
        "Árbol": DecisionTreeClassifier(random_state=42),                    # sin límite de profundidad (valores por defecto)
        "Random Forest": RandomForestClassifier(random_state=42),            # 100 árboles por defecto
        "Boosting": HistGradientBoostingClassifier(random_state=42),         # valores por defecto
    }


def modelo_elegido():
    """Modelo recomendado en la Tarea 6.2: boosting con pesos de clase balanceados.
    Aprobar un crédito malo cuesta más que rechazar uno bueno; con class_weight='balanced'
    cada error sobre un «bad» pesa más al entrenar (ver optimizacion.py y el informe)."""
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
