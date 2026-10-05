"""Paso 5 · Búsqueda de hiperparámetros, y la prueba honesta con datos que la búsqueda nunca vio."""
from sklearn.model_selection import GridSearchCV, train_test_split   # GridSearchCV: prueba todas las combinaciones de una cuadrícula

# El espacio de búsqueda: es parte del resultado, por eso se documenta. El prefijo 'modelo__' apunta al paso 'modelo' del pipeline.
PARAM_GRID = {
    "modelo__max_depth": [1, 2, 3, 4, 5, 7, 10, None],     # cuántas preguntas seguidas puede hacer el árbol (None = sin límite)
    "modelo__min_samples_leaf": [1, 2, 5, 10, 20],         # cuántos casos como mínimo debe tener cada hoja final
}


def buscar(pipeline, X, y, cv):
    """Prueba las 40 combinaciones con validación cruzada y devuelve la mejor y su puntaje."""
    # TODO 8 (en clase): GridSearchCV con PARAM_GRID
    raise NotImplementedError("TODO 8: completar en clase")


def buscar_con_prueba(pipeline, X, y, cv):
    """La búsqueda usa solo el 80%; el 20% reservado se abre una sola vez al final."""
    # TODO 9 (en clase): reservar el 20%, buscar con el 80% y evaluar una sola vez
    raise NotImplementedError("TODO 9: completar en clase")
