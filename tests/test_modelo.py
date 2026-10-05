"""Pruebas automáticas: si fallan, el cambio no se acepta. Se ejecutan con:  pytest -v"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # permite importar los módulos de la carpeta raíz
import pytest                                                        # pytest: encuentra y corre las funciones que empiezan con test_
from datos import cargar_datos
from modelos import crear_pipeline, tres_modelos
from validacion import validar

X, y, cols_cat, cv = cargar_datos()                                  # datos listos para todas las pruebas


@pytest.mark.parametrize("nombre", [
    pytest.param("Árbol", marks=pytest.mark.xfail(reason="el árbol sin límite no supera el 0.70")),  # se espera que falle: es la lección
    "Random Forest", "Boosting"])
def test_supera_a_siempre_bueno(nombre):
    # TODO 11 (en clase): assert que la media supera 0.70
    raise NotImplementedError("TODO 11: completar en clase")


def test_resultado_estable():
    # TODO 12 (en clase): assert que la desviación es menor que 0.05
    raise NotImplementedError("TODO 12: completar en clase")


def test_codificacion_dentro_del_pipeline():
    pipe = crear_pipeline(tres_modelos()["Árbol"], cols_cat)         # el pipeline que construimos
    assert "prep" in pipe.named_steps                                # la preparación debe ser un paso del pipeline (así no hay fuga)
