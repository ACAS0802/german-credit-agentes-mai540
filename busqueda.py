"""Paso 5 · Búsqueda de hiperparámetros, y la prueba honesta con datos que la búsqueda nunca vio."""
from sklearn.model_selection import GridSearchCV, train_test_split   # GridSearchCV: prueba todas las combinaciones de una cuadrícula

# El espacio de búsqueda: es parte del resultado, por eso se documenta. El prefijo 'modelo__' apunta al paso 'modelo' del pipeline.
PARAM_GRID = {
    "modelo__max_depth": [1, 2, 3, 4, 5, 7, 10, None],     # cuántas preguntas seguidas puede hacer el árbol (None = sin límite)
    "modelo__min_samples_leaf": [1, 2, 5, 10, 20],         # cuántos casos como mínimo debe tener cada hoja final
}


def buscar(pipeline, X, y, cv):
    """Prueba las 40 combinaciones con validación cruzada y devuelve la mejor y su puntaje."""
    gs = GridSearchCV(pipeline, PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X, y)                                       # 40 combinaciones × 5 pliegues
    return gs.best_params_, gs.best_score_             # ojo: best_score_ es optimista (eligió con esos mismos pliegues)


def buscar_con_prueba(pipeline, X, y, cv):
    """La búsqueda usa solo el 80%; el 20% reservado se abre una sola vez al final."""
    X_bus, X_res, y_bus, y_res = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)  # el 20% se aparta antes de buscar
    gs = GridSearchCV(pipeline, PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_bus, y_bus)                               # la búsqueda solo ve el 80%
    return gs.best_score_, gs.score(X_res, y_res)      # lo que dijo la búsqueda frente a la prueba reservada
