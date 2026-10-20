# Verificación de la respuesta a la tercera auditoría — rama `correccion/auditoria-ronda-3`

**Revisor:** Skill `auditoria-modelos` v5, sobre los commits de respuesta `3c81f90` (H1) y `aacc22c` (H2).
**Alcance:** solo las dos condiciones que fallaron en `AUDIT_REPORT_ronda3.md` (A6 y E4). El resto de la tabla no cambia: los commits de respuesta no tocan `datos.py`, `regla.py`, `seleccion.py` ni `busqueda.py`.
**Fecha:** 2026-10-20

| # | Antes | Ahora | Evidencia |
|---|---|---|---|
| A6 | FALLA | PASA | `capstone.py`, `auditar_elegido`: `s["procedimiento_anidado"], pred = evaluar_por_pliegues(boosting_afinado(cols_cat, CV_INTERNA, n_jobs=N_JOBS), X, y, cv)`. Las tablas de `resultados/subgrupos/` y `resultados/errores/` salen de esas predicciones, en las que el boosting se afina solo con el entrenamiento de cada pliegue externo. La versión con hiperparámetros fijos se guarda solo como referencia (`modelo_elegido_misma_cv`) |
| E4 | FALLA | PASA | `capstone.py` docstring: `"Las cifras que se reportan (pasos 2, 3, 4 y 6) son anidadas"` y `"Dos pasos usan los 1,000 casos a propósito y no producen cifras de desempeño: el 1 ... y el 5"`, lo que coincide con el código. `tests/test_auditoria.py`: `def test_modelo_elegido_es_el_que_eligio_la_regla()` existe y pasa |

*Medición propia de la auditoría*: las conclusiones del desglose coinciden con la medición propia de la tercera ronda. La mayor caída de un subgrupo concluyente es 0.031 (`housing = own`, p = 0.135), y con `no checking` el modelo detecta el 13 % de los malos. `pytest -q`: 11 passed, 1 xfailed.

**Resultado:** 24 PASA, 0 FALLA, 1 NO APLICA (C4). **Decisión:** aprobado.

Siguen vigentes, sin cambiar veredictos, las observaciones de la tercera ronda: el margen de exactitud sobre «siempre bueno» es mínimo, los hiperparámetros exactos no son estables, hay errores concentrados en `no checking` y en créditos cortos, y queda el uso indirecto de los atributos excluidos. Deben ir al README y al informe.
