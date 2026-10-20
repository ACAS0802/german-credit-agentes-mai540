# Auditoría (tercera ronda, Tarea 8.1) — german-credit-agentes-mai540, rama `correccion/auditoria-ronda-3`

**Archivos auditados:** `datos.py` (20 líneas), `modelos.py` (75), `regla.py` (40), `seleccion.py` (78), `busqueda.py` (24), `auditoria.py` (66), `capstone.py` (201); salidas: `resultados/capstone.json`, `resultados/capacidad_boosting.csv` (37 líneas), `resultados/subgrupos/*.csv` (5 archivos), `resultados/errores/*.csv` (4 archivos), `figs/capacidad_boosting.png`. Excluidos: `tests/`, `.claude/`, `entregas/`, `evidencia/`, `auditoria/` (informes anteriores); `optimizacion.py`, `experimentos.py`, `main.py`, `validacion.py` y `curvas.py` no producen ninguna cifra del capstone (ver *Lo que esta auditoría no revisó*).
**Modelo auditado:** el que elige la regla en `capstone.py`: `crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES)` con `HistGradientBoostingClassifier(class_weight="balanced", random_state=42, learning_rate=0.03, max_iter=100, max_leaf_nodes=8, min_samples_leaf=20)` (`modelos.py` líneas 73-75).
**Tipo de problema:** clasificación binaria (`datos.py` línea 9 crea un objetivo 0/1; todos los estimadores son `*Classifier`).
**Objetivo:** `clase` (y = 1 si `"bad"`)   **Clase de interés:** `"bad"` = 1 (aprobar un crédito malo cuesta 5 veces más que rechazar uno bueno)
**Columnas de subgrupo:** edad agrupada (19-24, 25-34, 35-49, 50+), sexo (`personal_status` agrupado), `personal_status`, `foreign_worker`, `housing`.
**Métrica prioritaria:** recall de `"bad"` (`malos_detectados`); decisión por costo 5:1; secundaria: exactitud.
**Umbral de disparidad:** 0.10 absoluto sobre el recall de `"bad"`, **justificado por el proyecto** (`auditoria.py` líneas 7-21: costo y ruido de muestreo).
**Convención de citas:** no hay notebooks; `archivo línea N` con líneas numeradas desde 1 (`cat -n`). Las mediciones propias se rotulan *medición propia de la auditoría* y se hicieron fuera del proyecto, sin escribir en él.
**Fecha:** 2026-10-20

## Resumen

De 25 verificaciones, 22 pasan, 2 fallan (A6 y E4) y 1 no aplica (C4). Los cuatro hallazgos de la segunda ronda quedaron corregidos: la regla es código y se aplica de forma anidada (`seleccion.py`), la capacidad del boosting se exploró con entrenamiento y validación, el umbral está argumentado por el costo y la robustez cubre a los siete candidatos. Las dos fallas que quedan no cambian ninguna conclusión según mi medición:

- **A6:** las tablas por subgrupo y por categoría salen de un modelo cuyos hiperparámetros se eligieron con los 1,000 casos, incluidos los pliegues que luego lo evalúan. Con predicciones totalmente anidadas las conclusiones son las mismas.
- **E4:** dos afirmaciones del código no son ciertas en esta rama: el docstring de `capstone.py` dice que *toda* búsqueda o elección es interna a un pliegue, y `modelos.py` cita una prueba que no existe.

Las cifras honestas del modelo son las anidadas: exactitud 0.704 ± 0.032, recall de «bad» 0.713, costo 640. Las de la selección anidada completa son 0.711 ± 0.030, 0.69 y 661.

## Tabla de verificación

| # | Verificación | Veredicto | Evidencia |
|---|---|---|---|
| A1 | Métricas en rango válido | PASA | `resultados/capstone.json`, `seleccion_anidada`: `"exactitud_media": 0.711`, `"malos_detectados": 0.69`; las exactitudes de `comparacion` van de 0.688 a 0.764 y los recalls de 0.357 a 0.713 |
| A2 | Métricas consistentes con la matriz de confusión | PASA | *Medición propia*: `seleccion_anidada` `"malos_aprobados": 93, "buenos_rechazados": 196` → recall (300−93)/300 = 0.69, exactitud 1 − 289/1000 = 0.711, costo 5·93 + 196 = 661, iguales a lo publicado. Lo mismo para `comparacion["Boosting balanceado afinado (8.1)"]` (86, 210 → 0.7133, 0.704, 640) y `modelo_elegido_misma_cv` (90, 200 → 0.70, 0.71, 650) |
| A3 | Con clases desbalanceadas se reporta más que la exactitud | PASA | `seleccion.py` líneas 72-74: `"malos_detectados"`, `"malos_aprobados"`, `"buenos_rechazados"`, `"costo"` |
| A4 | Métrica prioritaria según el costo, argumentada | PASA | `regla.py` líneas 7-9: `"Costo = 5 × (malos aprobados) + 1 × (buenos rechazados), la matriz de costos del German Credit original (Hofmann, 1994): aprobar un crédito malo cuesta 5 veces más que rechazar uno bueno."` |
| A5 | `pos_label` apunta a la clase de interés | PASA | `datos.py` línea 9: `y = (df["clase"] == "bad").astype(int)`; `regla.py` línea 20 y `seleccion.py` línea 65: `confusion_matrix(..., labels=[0, 1])`; `seleccion.py` línea 72: `tp / (tp + fn)`; `auditoria.py` línea 31: `t[t["real"] == 1]` |
| A6 | Ningún hiperparámetro se eligió con la partición que produce la métrica reportada | FALLA | Las cifras del modelo sí son anidadas (`capstone.py` línea 178: `evaluar_por_pliegues(SeleccionPorRegla(cands, CV_INTERNA), X, y, cv)`; línea 172 con el `GridSearchCV` interno). Pero las tablas por subgrupo y por categoría vienen de `capstone.py` líneas 128-129: `elegido = crear_pipeline(modelo_elegido(), ...)` / `evaluar_por_pliegues(elegido, X, y, cv)`, con hiperparámetros elegidos en la línea 185 sobre los 1,000 casos. Ver H1 |
| B1 | La partición ocurre antes de toda transformación que aprende | PASA | `modelos.py` líneas 18-26: `OrdinalEncoder` dentro de `Pipeline([("prep", prep), ("modelo", modelo)])`; `seleccion.py` línea 55: `clone(estimador).fit(X.iloc[ent], y.iloc[ent])` |
| B2 | Semilla fija en partición y estimadores | PASA | `datos.py` línea 12: `random_state=42`; `capstone.py` línea 38: `CV_INTERNA = StratifiedKFold(..., random_state=1)`; línea 105: `random_state=s`; `modelos.py` línea 73: `random_state=42`; candidatos en `capstone.py` líneas 46-55, todos con `random_state=42` |
| B3 | Partición estratificada | PASA | `datos.py` línea 12 y `capstone.py` líneas 38 y 105: `StratifiedKFold` |
| B4 | Los modelos comparados usan la misma partición y datos | PASA | `capstone.py` línea 172: los siete candidatos pasan por `evaluar_por_pliegues(e, X, y, cv)` con los mismos `X, y, cv`; línea 106 igual para cada semilla |
| B5 | La validación recibe un estimador sin entrenar | PASA | `seleccion.py` línea 55: `clone(estimador)` en cada pliegue; línea 31: `clone(self.candidatos[n])`; línea 37: `clone(...)` antes del ajuste final |
| B6 | Ninguna fila en entrenamiento y prueba a la vez | PASA | *Medición propia*: 0 filas duplicadas y 0 vectores de las 17 predictoras duplicados en 1,000 filas. Sin identificador de solicitante (ver *No revisó*) |
| C1 | Ningún transformador se ajusta con datos de prueba | PASA | `modelos.py` línea 26: `return Pipeline([("prep", prep), ("modelo", modelo)])`; el pipeline se ajusta solo con `X.iloc[ent]` (`seleccion.py` línea 55) |
| C2 | Ninguna predictora es posterior a la predicción | PASA | *Medición propia*: el pipeline ajustado expone 17 entradas (`checking_status, credit_history, purpose, savings_status, employment, other_parties, property_magnitude, other_payment_plans, housing, job, own_telephone, duration, credit_amount, installment_commitment, residence_since, existing_credits, num_dependents`), todas datos de la solicitud |
| C3 | Ninguna predictora separa el objetivo completo | PASA | Mismos datos y predictoras que la segunda ronda (solo `credit_amount` disparó la tabla cruzada, con exactitud 0.572 por sí sola). Descartada |
| C4 | Selección de características solo con entrenamiento | NO APLICA | La exclusión de `VARIABLES_SENSIBLES` (`modelos.py` línea 12) es una lista fija escrita a mano; no hay selección aprendida |
| C5 | El objetivo no está entre las predictoras | PASA | `datos.py` líneas 9-10: `y = ...` / `X = df.drop(columns="clase")` |
| D1 | Métrica prioritaria por subgrupo | PASA | `auditoria.py` línea 49: `"malos_detectados": r`; `capstone.py` líneas 132-134 la calculan para cinco columnas; `resultados/subgrupos/*.csv` |
| D2 | Ningún subgrupo con ≥ 30 malos cae más de 0.10 | PASA | `resultados/subgrupos/edad_grupo.csv`: mayor caída 0.0514 (35-49, 74 malos, p = 0.27). *Medición propia* con predicciones anidadas: mayor caída 0.031 (`housing = own`, 186 malos, p = 0.135) |
| D3 | El umbral está escrito y justificado | PASA | `auditoria.py` líneas 14-18: `"Una caída de 0.10 le quita a ese grupo 5 × 0.10 = 0.5 unidades por malo: cerca de un sexto del beneficio"`; líneas 19-21: error estándar ≈ 0.08 con 30 malos |
| D4 | Grupos pequeños como no concluyentes | PASA | `auditoria.py` líneas 45-47: `"no concluyente" if not concluyente`; `resultados/subgrupos/foreign_worker.csv`: `no`, 4 malos, `no concluyente` |
| E1 | Sin valores no finitos ni filas descartadas | PASA | *Medición propia*: 0 `NaN` en el CSV; `seleccion_anidada`: 93 + 196 + 207 + 504 = 1,000 predicciones |
| E2 | Reproducibilidad y consistencia interna | PASA | *Medición propia*: la reejecución completa de `capstone.py` (más `--auditoria`) en una copia limpia del código, 24 min, produce `resultados/capstone.json` idéntico clave por clave y los 10 CSV y `figs/capacidad_boosting.png` idénticos byte a byte. Dos ejecuciones de `evaluar_por_pliegues` del elegido dan diccionarios idénticos; la comparación reproduce exactamente los costos de la 6.2 (1058, 800, 944, 979, 853, 728) |
| E3 | Capacidad elegida con entrenamiento y validación | PASA | `modelos.py` líneas 45-50: `PARAM_GRID_BOOSTING` con `max_leaf_nodes`, `min_samples_leaf`, `learning_rate`; línea 61: `return_train_score=True`; `resultados/capacidad_boosting.csv` columnas `exactitud_entrenamiento`, `exactitud_validacion`, `brecha`; `figs/capacidad_boosting.png` |
| E4 | Afirmaciones respaldadas por salidas | FALLA | `capstone.py` líneas 11-12: `"Toda búsqueda o elección ocurre dentro de una validación interna (random_state=1) que solo ve los datos de entrenamiento de cada pliegue externo."` — las líneas 64 y 185 eligen con los 1,000 casos, y la 185 compara candidatos con `cv` (`random_state=42`). `modelos.py` línea 70: `"la prueba test_modelo_elegido_es_el_que_eligio_la_regla impide cambiarlos a mano"` — esa prueba no existe en `tests/`. Ver H2 |

## Hallazgos

### H1 — Las tablas por subgrupo y por categoría salen de un modelo afinado con todos los datos (A6)

**Qué encontré.** El modelo final se elige con los 1,000 casos (`capstone.py` línea 185: `final = SeleccionPorRegla(cands, cv).fit(X, y)`), lo que es correcto para entrenar la herramienta. Pero el paso 6 vuelve a evaluar ese modelo, con sus hiperparámetros ya fijos, sobre las mismas filas:

```python
    elegido = crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES)
    s["modelo_elegido_misma_cv"], pred = evaluar_por_pliegues(elegido, X, y, cv)
```

De esas predicciones salen todos los `resultados/subgrupos/*.csv` y `resultados/errores/*.csv`. Cada caso de prueba participó en la elección de `max_leaf_nodes=8` y `learning_rate=0.03`. El docstring de `auditoria_grupos` (`auditoria.py` líneas 27-28: `"cada caso lo predijo un modelo que no lo vio"`) es cierto para el ajuste, pero no para la elección de hiperparámetros.

*Medición propia de la auditoría*: con las predicciones fuera de pliegue del boosting afinado de forma anidada (búsqueda dentro de cada pliegue externo), las conclusiones no cambian. Ningún subgrupo concluyente supera 0.10 (máximo 0.031, `housing = own`), y siguen marcadas las mismas categorías de error: `no checking` (recall 0.13), créditos de hasta 12 meses (0.566). Además, en los pliegues externos la búsqueda nunca eligió 8 hojas: eligió 4 hojas en los cinco, con tasa 0.1 o 0.03.

**Por qué importa.** Es el mismo tipo de optimismo que A6 vigila, aplicado al desglose y no a la cifra global. Aquí el efecto medido es nulo, pero el proyecto no lo demuestra.

**Qué tan seguro estoy.** Confirmado por el código. El efecto nulo es una medición propia.

### H2 — Dos afirmaciones del código no son ciertas en esta rama (E4)

**Qué encontré.**
1. `capstone.py` líneas 11-12 afirman que *toda* búsqueda o elección ocurre dentro de una validación interna que solo ve el entrenamiento de cada pliegue externo. Las líneas 64 (exploración de capacidad) y 185 (elección final) eligen con los 1,000 casos, y la 185 compara a los candidatos con `cv`, no con `random_state=1`. Las dos son legítimas (una explora y la otra entrena la herramienta), pero la frase es más amplia que el código.
2. `modelos.py` línea 70 dice que `test_modelo_elegido_es_el_que_eligio_la_regla` impide cambiar los hiperparámetros a mano. `grep` en `tests/` no la encuentra: hoy nada comprueba que `modelo_elegido()` coincida con `resultados/capstone.json` (`eleccion_final.hiperparametros`). *Medición propia*: hoy coinciden.

**Por qué importa.** Un lector que confíe en el docstring creería que las tablas del paso 6 son anidadas (H1), y que un cambio manual del modelo lo detectarían las pruebas, cosa que no ocurre.

**Qué tan seguro estoy.** Confirmado por el código.

## Observaciones (no cambian veredictos)

- **El modelo cambia exactitud por costo.** Frente al de la 6.2 rechaza más buenos (196 a 210 frente a 128) para aprobar menos malos (86 a 93 frente a 120). Su exactitud honesta, 0.704, apenas supera 0.70: en las semillas 0 y 3 de la robustez queda en 0.696 y 0.695 y la regla elige el boosting de la 6.2. En las cinco semillas es, aun así, el de menor costo (591 a 672 frente a 727 a 760). El margen sobre «siempre bueno» en exactitud no es confiable; el ahorro de costo sí lo es. Conviene decirlo en el README y en el informe.
- **Los hiperparámetros exactos no son estables.** Varias combinaciones quedan a menos de 10 unidades de costo entre sí (`capacidad_boosting.csv`), y la búsqueda elige 8 hojas con `CV_INTERNA` pero 4 hojas con `cv` y en cada pliegue externo. Lo estable es la conclusión de fondo: pocas hojas (4 a 8) y tasa baja.
- **Errores concentrados en categorías no protegidas.** `resultados/errores/cuenta_corriente.csv`: con `no checking` el modelo detecta 9 % de los malos (46 malos); `duracion.csv`: hasta 12 meses, 58 % (76 malos). Son limitaciones de uso, no disparidades entre grupos protegidos, y deberían ir al README.
- **Uso indirecto de los atributos excluidos.** Sin cambios frente a la segunda ronda: las mismas 17 predictoras permiten reconstruir en parte los atributos excluidos (AUC 0.69-0.81).
- **Umbral de decisión.** El modelo usa `predict()` (0.5 con pesos balanceados). El umbral que implica la matriz 5:1 es 1/6; no se explora.

## Estado de los hallazgos de la segunda ronda

| Segunda ronda | Estado | Evidencia |
|---|---|---|
| H1 / A6 — el elegido se evalúa con las particiones que lo eligen | Corregido para las cifras globales | `seleccion.py` (`SeleccionPorRegla`); `capstone.py` línea 178. Queda el desglose (H1 de esta ronda) |
| H2 / E3 — capacidad no explorada | Corregido | `modelos.py` líneas 45-61; `capacidad_boosting.csv`; brecha 0.24 → 0.08 |
| H3 / D3 — umbral sin justificar | Corregido | `auditoria.py` líneas 7-21 |
| H4 / E4 — robustez con tres rivales | Corregido | `capstone.py` líneas 101-113: los siete candidatos en las cinco semillas, con `gana_por_regla` |
| Observación `supera_umbral = False` en grupos pequeños | Corregido | `auditoria.py` líneas 46-47: veredicto `"no concluyente"` |

## Acciones recomendadas

1. `capstone.py` paso 6 — Calcular las tablas por subgrupo y por categoría con las predicciones fuera de pliegue del procedimiento anidado (el `boosting_afinado` evaluado con `cv`) y rotular las del modelo fijo como referencia optimista. — Cierra A6 para el desglose.
2. `capstone.py` líneas 11-12 — Decir qué pasos son anidados (comparación, selección, robustez) y cuáles usan los 1,000 casos (exploración de capacidad, elección final), y por qué. — E4.
3. `tests/` — Agregar la prueba que `modelos.py` línea 70 promete: `modelo_elegido()` debe tener los hiperparámetros de `resultados/capstone.json` → `eleccion_final`. — E4.
4. README e informe — Reportar como cifras del modelo las anidadas (0.704 ± 0.032, 0.713, 640), decir que su exactitud apenas supera 0.70 y que el ahorro de costo es lo robusto, y listar `no checking` y créditos de hasta 12 meses como casos donde no sirve.

## Lo que esta auditoría no revisó

- **`optimizacion.py`, `experimentos.py`, `validacion.py`, `curvas.py`:** flujos de las Tareas 6.1 y 6.2 que no producen cifras del capstone. `optimizacion.py` línea 14 conserva la frase que la segunda ronda marcó (`"ningún modelo usa VARIABLES_SENSIBLES"`).
- **`main.py`:** sigue llamando a `crear_pipeline` sin `excluir`, es decir, con las variables sensibles. Es el flujo de la clase y no es la herramienta, pero quien lo ejecute verá otros números.
- **`tests/`**, salvo para comprobar si existe la prueba citada (H2).
- **Entidades repetidas (B6):** sin identificador de solicitante, solo se descartaron duplicados exactos.
- **C2:** depende de criterio de dominio (cuándo se captura cada campo).
- **Selección anidada con el boosting afinado dentro de otra búsqueda:** la medición propia de H1 usó el boosting afinado anidado, no la selección completa entre siete candidatos.
