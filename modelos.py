"""Paso 2 · Los modelos y el Pipeline: la codificación va DENTRO del pipeline para que nada se filtre entre pliegues."""
from sklearn.compose import ColumnTransformer          # ColumnTransformer: aplica una transformación solo a ciertas columnas
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # los dos modelos de conjunto
from sklearn.pipeline import Pipeline                  # Pipeline: encadena pasos (preparar → modelo) como una sola pieza
from sklearn.preprocessing import OrdinalEncoder       # OrdinalEncoder: convierte cada categoría de texto en un número
from sklearn.tree import DecisionTreeClassifier        # el árbol de decisión: nuestro modelo individual


# Atributos protegidos (edad, sexo/estado civil, origen nacional): se usan para auditar, no para decidir.
# Hallazgo E4 de la auditoría de la Tarea 6.2. 'housing' no es un atributo protegido: se queda como
# variable financiera, pero también se audita.
VARIABLES_SENSIBLES = ("age", "personal_status", "foreign_worker")


def crear_pipeline(modelo, cols_cat, excluir=()):
    """Envuelve un modelo con su preparación de datos. 'excluir' son columnas que el modelo no ve
    (la Tarea 6.1 no excluye nada; la 6.2 excluye VARIABLES_SENSIBLES)."""
    codificador = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)  # categoría no vista en entrenamiento → -1
    pasos = [("cat", codificador, [c for c in cols_cat if c not in excluir])]  # solo las columnas de texto se codifican
    if excluir:
        pasos.append(("fuera", "drop", list(excluir)))     # columnas que el modelo no puede usar
    prep = ColumnTransformer(
        pasos,
        remainder="passthrough",                           # las numéricas pasan tal cual (los árboles no necesitan escalado)
    )
    return Pipeline([("prep", prep), ("modelo", modelo)])  # 'prep' se ajusta solo con los datos de entrenamiento de cada pliegue


def tres_modelos():
    """Los tres modelos que comparamos, con semilla fija."""
    return {
        "Árbol": DecisionTreeClassifier(random_state=42),                    # sin límite de profundidad (valores por defecto)
        "Random Forest": RandomForestClassifier(random_state=42),            # 100 árboles por defecto
        "Boosting": HistGradientBoostingClassifier(random_state=42),         # valores por defecto
    }


# Tarea 8.1 · Capacidad del boosting (hallazgo H2 / E3). En la 6.2 el boosting quedó con sus valores por
# omisión y su curva mostraba 0.996 en entrenamiento frente a 0.752 en validación: casi memorizaba.
# Se exploran los hiperparámetros que controlan cuánto puede aprender:
#   max_leaf_nodes   hojas por árbol (31 por omisión): cuántas reglas puede combinar cada árbol
#   min_samples_leaf casos mínimos por hoja (20 por omisión): evita reglas basadas en pocos casos
#   learning_rate    cuánto corrige cada árbol nuevo (0.1 por omisión)
#   max_iter         número de árboles (100 por omisión)
PARAM_GRID_BOOSTING = {
    "modelo__max_leaf_nodes": [4, 8, 31],
    "modelo__min_samples_leaf": [20, 50],
    "modelo__learning_rate": [0.03, 0.1],
    "modelo__max_iter": [50, 100, 300],
}


def boosting_afinado(cols_cat, cv_interna, n_jobs=1):
    """Boosting balanceado cuya capacidad se elige con la regla del proyecto (regla.py) dentro de
    cv_interna. Reporta entrenamiento y validación (return_train_score) para ver dónde memoriza."""
    from sklearn.model_selection import GridSearchCV
    from regla import METRICAS, refit_por_regla
    base = crear_pipeline(HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
                          cols_cat, VARIABLES_SENSIBLES)
    return GridSearchCV(base, PARAM_GRID_BOOSTING, cv=cv_interna, scoring=METRICAS,
                        refit=refit_por_regla, return_train_score=True, n_jobs=n_jobs)


def modelo_elegido():
    """Modelo de la herramienta (Tarea 8.1): el que eligió la regla de regla.py en capstone.py.

    Boosting con pesos de clase balanceados (aprobar un malo cuesta 5 veces más, así que cada error
    sobre un «bad» pesa más al entrenar) y con capacidad reducida frente al de la 6.2: árboles de 8
    hojas en vez de 31 y tasa de aprendizaje 0.03 en vez de 0.1. Así deja de memorizar el
    entrenamiento (brecha 0.08 frente a 0.24). Los valores salen de resultados/capstone.json
    («eleccion_final»); la prueba test_modelo_elegido_es_el_que_eligio_la_regla impide cambiarlos a mano.
    El modelo de la 6.2 era HistGradientBoostingClassifier(class_weight="balanced", random_state=42)."""
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=42,
                                          learning_rate=0.03, max_iter=100,
                                          max_leaf_nodes=8, min_samples_leaf=20)
