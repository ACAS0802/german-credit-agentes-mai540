"""Pruebas de la regla de elección y de la auditoría por subgrupo (Tarea 8.1).  pytest -v"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auditoria import MIN_MALOS, auditar_grupos
from modelos import modelo_elegido
from regla import indice_por_regla

RAIZ = os.path.join(os.path.dirname(__file__), "..")


def test_regla_elige_el_menor_costo_entre_los_que_superan_070():
    # el tercero es el más barato, pero no supera a «siempre bueno»: no puede ganar
    assert indice_por_regla([0.75, 0.72, 0.69], [944, 728, 500]) == 1


def test_modelo_elegido_es_el_que_eligio_la_regla():
    # Si alguien cambia modelos.py a mano, esta prueba lo detecta: el modelo de la herramienta debe
    # ser el que la regla eligió en capstone.py (resultados/capstone.json, «eleccion_final»).
    with open(os.path.join(RAIZ, "resultados", "capstone.json"), encoding="utf-8") as f:
        final = json.load(f)["eleccion_final"]
    params = modelo_elegido().get_params()
    for k, v in final["hiperparametros"].items():
        assert params[k] == v, f"{k}: modelos.py tiene {params[k]}, la regla eligió {v}"


def test_grupo_pequeno_es_no_concluyente_aunque_caiga_mucho():
    # 10 malos en el grupo «b», ninguno detectado: la caída es enorme, pero 10 < 30 malos.
    y = [1] * 110
    pred = [1] * 100 + [0] * 10
    tabla = auditar_grupos(y, pred, ["a"] * 100 + ["b"] * 10)
    assert tabla.loc["b", "malos_en_grupo"] < MIN_MALOS
    assert tabla.loc["b", "veredicto"] == "no concluyente"


def test_auditoria_detecta_una_disparidad_real():
    # El verificador debe seguir detectando lo que debe: 50 malos con solo la mitad detectados.
    y = [1] * 150
    pred = [1] * 100 + [1] * 25 + [0] * 25
    tabla = auditar_grupos(y, pred, ["a"] * 100 + ["b"] * 50)
    assert tabla.loc["b", "veredicto"] == "supera el umbral"
