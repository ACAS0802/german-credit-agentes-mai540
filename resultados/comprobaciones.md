# Comprobaciones antes de entregar (Tarea 8.1)

Las tres comprobaciones del Módulo 8, con el resultado de cada una. Fecha: 20 de octubre de 2026.

| Comprobación | Resultado | Registro completo |
|---|---|---|
| 1. Pruebas automáticas (`pytest -v`) | 18 passed, 1 xfailed (el fallo esperado es el árbol sin límite de la 6.1) | `resultados/pytest.txt` |
| 2. Hook de Git (`hooks/pre-commit`, del Módulo 7) | Cancela el commit cuando una prueba falla y lo deja pasar cuando todas pasan | Este archivo, sección 2 |
| 3. Auditoría con la Skill (fuga, partición y subgrupos) | 24 PASA, 0 FALLA, 1 NO APLICA | `auditoria/AUDIT_REPORT_ronda3.md` y `auditoria/AUDIT_REPORT_ronda3_verificacion.md` |

## 1. Pruebas automáticas

```
$ pytest -q
...........x.......
18 passed, 1 xfailed
```

## 2. Hook de Git

Instalado con `python instalar_hook.py`. La prueba se hizo en una copia del repositorio para no dejar commits de prueba en el historial.

```
## 1. Commit cancelado: alguien cambia el modelo a mano (max_leaf_nodes=8 → 31, el valor de la 6.2)

$ git commit -m "Prueba: vuelve a 31 hojas"
Ejecutando las pruebas antes del commit...
E           AssertionError: max_leaf_nodes: modelos.py tiene 31, la regla eligió 8
FAILED tests/test_auditoria.py::test_modelo_elegido_es_el_que_eligio_la_regla
1 failed, 17 passed, 1 xfailed in 6.00s
Las pruebas fallaron: commit cancelado.
$ git log --oneline -1      # el commit no se guardó
de21f17 Guarda la salida de pytest -v con todas las pruebas: 18 passed, 1 xfailed

## 2. Commit aceptado: con el modelo que eligió la regla, las pruebas pasan

$ git commit -m "Prueba: commit con las pruebas en verde"
Ejecutando las pruebas antes del commit...
18 passed, 1 xfailed in 4.62s
Todo en orden: el commit continúa.
[funcionalidad/readme d8f8ddf] Prueba: commit con las pruebas en verde
```

El caso 1 es el que el hook debe atrapar: `test_modelo_elegido_es_el_que_eligio_la_regla` impide volver al boosting de 31 hojas de la 6.2 (el que memorizaba) sin pasar por la regla de elección.

## 3. Auditoría

| Revisión | Qué encontró | Qué se corrigió | Cómo se comprobó |
|---|---|---|---|
| Fuga de datos | Ninguna: la codificación está dentro del pipeline; 0 duplicados; ninguna variable es posterior a la decisión | — | Verificaciones B1, C1-C5 en PASA |
| Partición | Las tablas por subgrupo usaban un modelo afinado con todos los datos (A6) | Predicciones del procedimiento anidado (`capstone.py`, paso 6) | A6 en PASA en la verificación; mismas conclusiones |
| Subgrupos | Ningún grupo protegido con ≥ 30 malos cae más de 0.10 (máximo 0.031) | Umbral justificado por el costo; «no concluyente» para grupos pequeños | D1-D4 en PASA |
