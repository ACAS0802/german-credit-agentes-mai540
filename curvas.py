"""Paso 4 · Curva de aprendizaje: ¿mejora el modelo con más datos? ¿hay sobreajuste?"""
import numpy as np
import matplotlib
matplotlib.use("Agg")                                   # modo sin ventana: guarda la gráfica en un archivo
import matplotlib.pyplot as plt                         # matplotlib: la librería para hacer gráficas
from sklearn.model_selection import learning_curve      # learning_curve: entrena con cantidades crecientes de datos


def curva_aprendizaje(pipeline, X, y, cv):
    """Devuelve los tamaños de entrenamiento y la exactitud media en entrenamiento y en validación."""
    # TODO 7 (en clase): learning_curve con train_sizes y promedio por pliegue
    raise NotImplementedError("TODO 7: completar en clase")


def graficar(tamanos, ent, val, titulo, ruta):
    """Dibuja la curva y la guarda como imagen."""
    plt.figure(figsize=(5, 3.5))                       # lienzo de 5 x 3.5 pulgadas
    plt.plot(tamanos, ent, "o-", label="entrenamiento")  # línea de la exactitud en entrenamiento
    plt.plot(tamanos, val, "o-", label="validación")     # línea de la exactitud en validación
    plt.axhline(0.70, ls="--", color="gray")             # referencia: 'siempre bueno' acierta el 70%
    plt.xlabel("casos de entrenamiento"); plt.ylabel("exactitud"); plt.title(titulo); plt.legend()  # etiquetas y leyenda
    plt.tight_layout(); plt.savefig(ruta, dpi=150); plt.close()  # ajusta márgenes, guarda y cierra
