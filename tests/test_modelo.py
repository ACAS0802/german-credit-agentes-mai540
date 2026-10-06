"""Pruebas automáticas: si fallan, el cambio no se acepta. Se ejecutan con:  pytest -v"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # permite importar los módulos de la carpeta raíz
import pytest                                                        # pytest: encuentra y corre las funciones que empiezan con test_
from datos import cargar_datos
from modelos import VARIABLES_SENSIBLES, crear_pipeline, modelo_elegido, tres_modelos
from validacion import validar

X, y, cols_cat, cv = cargar_datos()                                  # datos listos para todas las pruebas


@pytest.mark.parametrize("nombre", [
    pytest.param("Árbol", marks=pytest.mark.xfail(reason="el árbol sin límite no supera el 0.70")),  # se espera que falle: es la lección
    "Random Forest", "Boosting"])
def test_supera_a_siempre_bueno(nombre):
    media, _ = validar(crear_pipeline(tres_modelos()[nombre], cols_cat), X, y, cv)
    assert media > 0.70, f"{nombre}: {media:.3f} no supera a «siempre bueno» (0.70)"


def test_resultado_estable():
    # Umbral 0.05: el modelo elegido supera a «siempre bueno» por 0.052 (0.752 frente a 0.70).
    # Con una desviación mayor que esa distancia, un pliegue típico podría quedar por debajo de la
    # referencia y la ventaja dejaría de ser confiable.
    _, desv = validar(crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES), X, y, cv)
    assert desv < 0.05, f"desviación {desv:.3f} ≥ 0.05"


def test_codificacion_dentro_del_pipeline():
    pipe = crear_pipeline(tres_modelos()["Árbol"], cols_cat)         # el pipeline que construimos
    assert "prep" in pipe.named_steps                                # la preparación debe ser un paso del pipeline (así no hay fuga)


# --- Pruebas del modelo elegido en la Tarea 6.2 -------------------------------------------------
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score

ELEGIDO = crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES)


def test_elegido_supera_a_siempre_bueno():
    media, _ = validar(ELEGIDO, X, y, cv)
    assert media > 0.70


def test_elegido_detecta_al_menos_la_mitad_de_los_malos():
    # El error caro es aprobar un crédito malo: el modelo elegido debe detectar al menos la mitad.
    pred = cross_val_predict(ELEGIDO, X, y, cv=cv)
    assert recall_score(y, pred) >= 0.50


def test_variables_sensibles_no_deciden():
    # Las variables sensibles se usan para auditar, no para decidir: si se barajan, las
    # predicciones del modelo entrenado no pueden cambiar.
    modelo = ELEGIDO.fit(X, y)
    barajado = X.copy()
    rng = np.random.default_rng(0)
    for col in VARIABLES_SENSIBLES:
        barajado[col] = rng.permutation(barajado[col].values)
    assert (modelo.predict(X) == modelo.predict(barajado)).all()
