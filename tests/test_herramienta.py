"""Pruebas de la herramienta (Tarea 8.1): que responda bien y que rechace lo que no debe aceptar.  pytest -v"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest

from herramienta import EJEMPLO, evaluar_solicitud


# --- La herramienta -----------------------------------------------------------------------------

def test_devuelve_probabilidad_valida_y_limitaciones():
    r = evaluar_solicitud(EJEMPLO)
    assert 0.0 <= r["probabilidad_malo"] <= 1.0
    assert r["riesgo"] in ("alto", "bajo")
    assert "no debe usarse" in r["limitaciones"].lower()         # quien la usa lee para qué no sirve


@pytest.mark.parametrize("cambio, mensaje", [
    ({"duration": -6}, "duration"),                              # una duración negativa no existe
    ({"credit_amount": 0}, "credit_amount"),                     # un crédito de 0 no existe
    ({"purpose": "vacaciones"}, "purpose"),                      # categoría que el modelo nunca vio
    ({"duration": "doce"}, "número"),                            # texto donde va un número
])
def test_rechaza_entrada_invalida(cambio, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        evaluar_solicitud({**EJEMPLO, **cambio})


def test_rechaza_solicitud_incompleta():
    incompleta = {k: v for k, v in EJEMPLO.items() if k != "checking_status"}
    with pytest.raises(ValueError, match="faltan datos: checking_status"):
        evaluar_solicitud(incompleta)


def test_edad_sexo_y_origen_no_cambian_la_respuesta():
    # Se usan para auditar, no para decidir: enviarlos con cualquier valor da el mismo resultado.
    a = evaluar_solicitud({**EJEMPLO, "age": 22, "personal_status": "female div/dep/mar", "foreign_worker": "yes"})
    b = evaluar_solicitud({**EJEMPLO, "age": 67, "personal_status": "male single", "foreign_worker": "no"})
    assert a["probabilidad_malo"] == b["probabilidad_malo"]


# --- La interfaz (app_gradio.py) ------------------------------------------------------------------

def test_interfaz_muestra_error_en_vez_de_fallar():
    # Quien usa la app no debe ver un error de Python: una duración de 0 se explica en pantalla.
    import app_gradio as app
    valores = [EJEMPLO[c] for c in app.CAMPOS]
    valores[app.CAMPOS.index("duration")] = 0
    assert "no válida" in app.evaluar(*valores)
    assert "Riesgo alto" in app.evaluar(*[EJEMPLO[c] for c in app.CAMPOS])
