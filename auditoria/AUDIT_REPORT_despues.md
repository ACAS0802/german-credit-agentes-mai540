# Auditoría (segunda ronda, después de las correcciones) — german-credit-agentes-mai540 (Tarea 6.2, modelo recomendado)

**Archivos auditados:** `datos.py` (20 líneas), `modelos.py` (42), `validacion.py` (28), `busqueda.py` (24), `curvas.py` (25), `auditoria.py` (17), `optimizacion.py` (207); salidas regeneradas: `resultados/optimizacion.json` (536 líneas), `resultados/busqueda_arbol.csv` (41), `resultados/subgrupos_edad_grupo.csv` (5), `resultados/subgrupos_personal_status.csv` (5), `resultados/subgrupos_housing.csv` (4), `resultados/subgrupos_foreign_worker.csv` (3), `figs/curvas_6_1_vs_6_2.png`. Excluidos por indicación del dueño: `tests/`, `.claude/`, `entregas/`, `evidencia/`, `auditoria/` (informe de la primera ronda). `experimentos.py` (148 líneas, sin cambios) se excluyó por irrelevante: solo evalúa el árbol y el Random Forest de la 6.1.
**Modelo auditado:** `crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES)` — `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)` sin `age`, `personal_status` ni `foreign_worker` — evaluado con la validación cruzada externa de `datos.py` (`StratifiedKFold(5, shuffle=True, random_state=42)`).
**Tipo de problema:** clasificación binaria (declarado por el dueño; coincide con el código: `datos.py` línea 9 crea un objetivo 0/1 y los estimadores son `*Classifier`).
**Objetivo:** `clase` (y = 1 si `"bad"`)   **Clase de interés:** `"bad"` = 1 (aprobar un crédito malo cuesta 5 veces más que rechazar uno bueno)
**Columnas de subgrupo:** edad agrupada (19-24, 25-34, 35-49, 50+), `personal_status`, `housing`, `foreign_worker` — declaradas por el dueño como columnas solo de auditoría.
**Métrica prioritaria:** recall de `"bad"` (`malos_detectados`); secundaria: exactitud.
**Umbral de disparidad:** 0.10 absoluto sobre el recall de `"bad"`. Es el valor por omisión de la Skill; el proyecto lo escribe ahora (`optimizacion.py` líneas 21-23, 46), pero lo justifica solo como «valor por omisión de la Skill».
**Convención de citas:** no hay notebooks; las citas son `archivo línea N`, con líneas numeradas desde 1 tal como las muestra `cat -n`. Las mediciones propias se rotulan *medición propia de la auditoría*. Sus scripts están en `/tmp/claude-0/skill62/ronda2/`, y la reejecución completa de `optimizacion.py` se hizo sobre una copia del código y los datos en `/tmp/claude-0/skill62/ronda2/copia/`, sin escribir nada dentro del proyecto.
**Fecha:** 2026-10-06

## Resumen

De 25 verificaciones, 20 pasan, 4 fallan (A6, D3, E3, E4) y 1 no aplica (C4); ninguna queda en `NO SE PUEDE DETERMINAR`. Frente a la primera ronda se corrigieron D4 y la parte grave de E4: el modelo ya no recibe edad, sexo/estado civil ni condición de trabajador extranjero, lo comprobé en el pipeline ajustado. La regla de elección y el umbral están ahora escritos. La reejecución completa de `optimizacion.py` reproduce todas las salidas (exactitud 0.752 ± 0.031, recall de «bad» 0.60, costo 728), así que las métricas se pueden creer como estimación de validación cruzada. Lo que queda abierto:

- **A6:** el elegido sigue evaluándose con las particiones que lo eligen. Aplicada de forma anidada, la regla escrita da el mismo modelo y las mismas cifras.
- **E3:** la capacidad del boosting sigue sin explorarse (entrenamiento 0.996 frente a validación 0.752).
- **D3:** el umbral está escrito, pero no justificado por el costo del proyecto.
- **E4:** la confirmación de la elección en las semillas 0-4 omite dos de los cinco rivales. Es cierta en sustancia según mi medición, pero no la respalda una salida del proyecto.

Ningún subgrupo con al menos 30 malos cae más de 0.10 (máximo 0.041, `housing = own`).

## Tabla de verificación

| # | Verificación | Veredicto | Evidencia |
|---|---|---|---|
| A1 | Métricas en rango válido | PASA | `resultados/optimizacion.json` líneas 199-214 (elegido): `"exactitud_media": 0.752`, `"malos_detectados": 0.6`; todas las exactitudes y recalls de `comparacion` están entre 0.3567 y 0.785 |
| A2 | Métricas consistentes con la matriz de confusión | PASA | `optimizacion.json` líneas 209-211: `"malos_aprobados": 120, "buenos_rechazados": 128, "costo": 728`. Recalculado: recall = (300−120)/300 = 0.60; exactitud = 1 − 248/1000 = 0.752; costo = 5·120 + 128 = 728. Las otras cinco filas de `comparacion` cuadran igual (p. ej. árbol afinado balanceado, líneas 167-181: 178/300 = 0.5933, 1 − 312/1000 = 0.688, 5·122 + 190 = 800) |
| A3 | Con clases desbalanceadas se reporta más que la exactitud | PASA | 30 % de «bad» (`datos.py` línea 19); `optimizacion.py` líneas 61-63 reportan `"malos_detectados"`, `"malos_aprobados"`, `"buenos_rechazados"`, `"costo"` |
| A4 | La métrica prioritaria corresponde al costo del error y está argumentada | PASA | `optimizacion.py` líneas 11-12: `"Costo: 5 × (malos aprobados) + 1 × (buenos rechazados), la matriz de costos del dataset original (Hofmann, 1994). Aprobar un crédito malo cuesta 5 veces más."`; líneas 17-18: la regla de elección usa ese costo (`"el de menor costo"`) |
| A5 | `pos_label` apunta a la clase de interés en toda métrica binaria | PASA | `datos.py` línea 9: `y = (df["clase"] == "bad").astype(int)`; `optimizacion.py` línea 61: `recall_score(y, pred)` (por omisión `pos_label=1` = «bad»); línea 167: `recall_score(y, cross_val_predict(elegido, X, y, cv=cv))`; línea 56: `tn, fp, fn, tp = confusion_matrix(y, pred).ravel()`; `validacion.py` línea 17: `pos_label=1`; `auditoria.py` línea 11: `tabla["real"] == 1` |
| A6 | Ningún hiperparámetro se eligió con la partición que produce la métrica reportada | FALLA | `optimizacion.py` líneas 17-19: `"Regla de elección (hallazgo A6): entre los modelos cuya exactitud media supera 0.70 («siempre bueno»), el de menor costo; la elección se confirma solo si ese modelo tiene el menor costo también en las 5 semillas adicionales de la validación cruzada (0 a 4)."` La regla se aplica sobre `comparacion` (líneas 111-123), calculada con la misma `cv` de la que salen las cifras reportadas del elegido (línea 123: `evaluar(crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES), X, y, cv)`); no hay evaluación anidada de la regla ni prueba reservada para el elegido. Ver H1 |
| B1 | La partición ocurre antes de toda transformación que aprende | PASA | `modelos.py` líneas 18-26: `OrdinalEncoder` y `("fuera", "drop", list(excluir))` dentro de `Pipeline([("prep", prep), ("modelo", modelo)])`; `optimizacion.py` líneas 54-55: `cross_validate` / `cross_val_predict` ajustan el pipeline completo en cada pliegue |
| B2 | Semilla fija en partición y estimadores | PASA | `datos.py` línea 12: `random_state=42`; `modelos.py` línea 42: `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)`; `optimizacion.py` línea 47: `CV_INTERNA = StratifiedKFold(..., random_state=1)`; línea 135: `StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)`; `busqueda.py` línea 21 y `curvas.py` línea 14: `random_state=42` |
| B3 | Partición estratificada | PASA | `datos.py` línea 12: `StratifiedKFold(...)`; `busqueda.py` línea 21: `stratify=y` |
| B4 | Los modelos comparados usan la misma partición y datos | PASA | `optimizacion.py` líneas 111-123: todos los candidatos pasan por `evaluar(..., X, y, cv)` con `X, y, cv` de la línea 69, y todos con `excluir=VARIABLES_SENSIBLES` (líneas 82, 113, 114, 117, 122, 123) |
| B5 | La validación cruzada recibe un estimador sin entrenar | PASA | `optimizacion.py` línea 123: el pipeline se construye en la misma llamada a `evaluar`; `evaluar` (líneas 54-55) solo llama a funciones que clonan. El `.fit` de la línea 95 (`gs_anidado.fit(...)`) ocurre después de evaluar `gs_anidado` (línea 94) |
| B6 | Ninguna fila en entrenamiento y prueba a la vez | PASA | *Medición propia de la auditoría* (primera ronda, sin cambios en `datos/credit_g.csv`): 0 filas duplicadas y 0 vectores de predictoras duplicados en 1,000 filas; con 17 predictoras tampoco hay duplicados que afecten (el conjunto no cambió) |
| C1 | Ningún transformador se ajusta con datos de prueba | PASA | `modelos.py` línea 26: `return Pipeline([("prep", prep), ("modelo", modelo)])  # 'prep' se ajusta solo con los datos de entrenamiento de cada pliegue` |
| C2 | Ninguna predictora es posterior a la predicción | PASA | *Medición propia de la auditoría*: el pipeline ajustado expone 17 entradas, `get_feature_names_out()` = `checking_status, credit_history, purpose, savings_status, employment, other_parties, property_magnitude, other_payment_plans, housing, job, own_telephone, duration, credit_amount, installment_commitment, residence_since, existing_credits, num_dependents`. Todas son datos de la solicitud; ningún nombre del tipo `*_final`, `*_programad*`, `resultado_*` |
| C3 | Ninguna predictora separa el objetivo completo | PASA | *Medición propia de la auditoría* (primera ronda, mismos datos): solo `credit_amount` disparó la tabla cruzada por su cardinalidad (921 valores); un árbol de una sola variable con validación cruzada da 0.572, menos que el 0.70 de «siempre bueno». Descartada |
| C4 | Selección de características ajustada solo con entrenamiento | NO APLICA | No hay selección basada en datos: la exclusión de `VARIABLES_SENSIBLES` (`modelos.py` línea 12) es una lista fija escrita a mano, no aprendida; no hay `SelectKBest`, `mutual_info` ni filtros por correlación |
| C5 | El objetivo no está entre las predictoras | PASA | `datos.py` líneas 9-10: `y = (df["clase"] == "bad").astype(int)` / `X = df.drop(columns="clase")` |
| D1 | Métrica prioritaria calculada por subgrupo | PASA | `auditoria.py` línea 16: `"malos_detectados": ... # recall de «bad» dentro del grupo`; `optimizacion.py` líneas 164-172 la calculan para las cuatro columnas y añaden `caida_vs_global` |
| D2 | Ningún subgrupo con ≥ 30 malos cae más de 0.10 bajo el global | PASA | *Medición propia de la auditoría* (recall global 0.60): mayor caída entre subgrupos con ≥ 30 malos = 0.0409 (`housing = own`, 186 malos, p = 0.070); le sigue 0.0324 (edad 35-49). Coincide con el proyecto: `resultados/subgrupos_housing.csv` línea 3: `own,713,0.7616,0.2609,0.5591,186,0.0409,True,False` |
| D3 | El umbral está escrito y justificado en el proyecto | FALLA | Escrito: `optimizacion.py` línea 46: `UMBRAL_DISPARIDAD, MIN_MALOS = 0.10, 30`. Justificación, líneas 21-22: `"caída de 0.10 absoluto en malos detectados de un subgrupo frente al global (valor por omisión de la Skill)"`. La Skill exige reemplazar el valor por omisión por un umbral argumentado según el costo del error; el proyecto no da ese argumento. Ver H3 |
| D4 | Subgrupos con < 30 casos reportados como no concluyentes | PASA | `optimizacion.py` líneas 168-171: `tabla["malos_en_grupo"] = ...`, `tabla["concluyente"] = tabla["malos_en_grupo"] >= MIN_MALOS`; `resultados/subgrupos_foreign_worker.csv` línea 2: `no,37,0.8108,0.1081,0.5,4,0.1,False,False`; `subgrupos_personal_status.csv` línea 4: `male mar/wid,92,0.6957,0.2717,0.48,25,0.12,False,False` y línea 3: `male div/sep,50,0.66,0.4,0.65,20,-0.05,False,False` |
| E1 | Sin valores no finitos ni filas descartadas | PASA | *Medición propia de la auditoría*: 0 `NaN` y 0 `inf` en el CSV; `optimizacion.json` (elegido): 572 + 128 + 120 + 180 = 1,000 predicciones (reconstruido de `malos_aprobados`, `buenos_rechazados` y 300 malos) |
| E2 | Reproducibilidad y consistencia interna | PASA | *Medición propia de la auditoría*: la reejecución completa de `optimizacion.py` sobre una copia (2,564 s) produce `busqueda_arbol.csv` y los cuatro `subgrupos_*.csv` idénticos byte a byte, y un `optimizacion.json` idéntico salvo `tiempo_ajuste_s`. La medición independiente del elegido da pliegues `[0.775, 0.725, 0.715, 0.76, 0.785]`, `tn fp fn tp = 572 128 120 180`; dos ejecuciones de `cross_val_predict` son idénticas |
| E3 | Capacidad elegida con entrenamiento y validación reportados | FALLA | `modelos.py` línea 42: `return HistGradientBoostingClassifier(class_weight="balanced", random_state=42)` — `max_iter`, `max_leaf_nodes`, `max_depth`, `learning_rate` y `min_samples_leaf` siguen en sus valores por omisión, sin exploración; `PARAM_GRID` (`busqueda.py` líneas 5-8) solo se aplica al árbol. Síntoma: `optimizacion.json` líneas 362-394, entrenamiento 0.9963 frente a validación 0.752, `"brecha_final": 0.2443`. Ver H2 |
| E4 | Afirmaciones respaldadas por salidas, salidas interpretadas | FALLA | La afirmación de la primera ronda ya es cierta (ver H4). Lo que queda: `optimizacion.py` líneas 18-19 dicen que la elección `"se confirma solo si ese modelo tiene el menor costo también en las 5 semillas adicionales"`, pero `robustez` (líneas 128-133) solo incluye a `"Random Forest"`, `"Boosting"`, `"Árbol afinado balanceado (anidada)"` y `"Boosting balanceado"`; faltan `"Árbol afinado (anidada)"` (exactitud 0.714 > 0.70, elegible) y `"Random Forest balanceado"` (0.757, elegible). Además, línea 14: `"ningún modelo usa VARIABLES_SENSIBLES"`, mientras la línea 144 entrena el árbol de la curva con `crear_pipeline(DecisionTreeClassifier(random_state=42), cols_cat)  # tal como en la 6.1`. Ver H4 |

## Hallazgos

### H1 — El modelo elegido sigue evaluándose con las particiones que lo eligen (A6)

**Qué encontré.** La regla de elección está escrita (`optimizacion.py` líneas 17-19) y es aplicable: con `comparacion` (líneas 111-123), los modelos con exactitud > 0.70 y sus costos son Árbol afinado (anidada) 1058, Random Forest 944, Boosting 853, Random Forest balanceado 979 y Boosting balanceado 728. Gana el elegido; el árbol balanceado (0.688, costo 800) queda fuera por exactitud y también por costo. Pero la regla se aplica a cifras calculadas con la misma `cv` de `datos.py` que luego se reporta como desempeño del elegido:

```python
    comparacion["Boosting balanceado (elegido)"] = evaluar(crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES), X, y, cv)
```

La confirmación con las semillas 0-4 (líneas 134-137) usa otras particiones de los mismos 1,000 casos, no datos que la elección no haya visto. El código no ejecuta la regla: `modelo_elegido()` está fijado a mano (`modelos.py` línea 42).

*Medición propia de la auditoría*: apliqué la regla escrita de forma anidada (exactitud > 0.70 y menor costo, medidos con una CV interna `random_state=1` dentro de cada pliegue externo, entre Random Forest, Random Forest balanceado, Boosting y Boosting balanceado). Elige Boosting balanceado en los 5 pliegues y reproduce exactamente exactitud 0.752, recall 0.60 y costo 728. Los dos árboles con búsqueda anidada no entraron a esta medición por su costo de cómputo; en la `cv` principal y en las cinco semillas su costo es mayor que el del elegido (800-1058 frente a 727-760).

**Por qué importa.** El procedimiento sigue permitiendo optimismo por selección: las cifras del elegido son las del ganador entre seis candidatos vistos en las mismas particiones. En este caso la medición propia indica que el optimismo es nulo, pero el proyecto no lo demuestra.

**Qué tan seguro estoy.** Confirmado por el código (la selección y la evaluación comparten partición). El efecto nulo es una medición propia, que no cubre a los dos árboles con búsqueda anidada.

### H2 — La capacidad del boosting sigue sin explorarse (E3)

**Qué encontré.** Sin cambios respecto de la primera ronda: `modelos.py` línea 42 deja `max_iter` (número de árboles del boosting), `max_leaf_nodes` (31), `max_depth` (None), `learning_rate` (0.1) y `min_samples_leaf` (20) en sus valores por omisión, y ningún archivo los explora. La curva de `curvas.py` varía el tamaño de entrenamiento, no la capacidad. El síntoma persiste: `optimizacion.json` líneas 362-394, entrenamiento 0.9963 frente a validación 0.752 (brecha 0.2443).

**Por qué importa.** El modelo casi memoriza el entrenamiento. Sin una exploración de capacidad con entrenamiento y validación juntos no se sabe si 0.752 / 0.60 / 728 es lo que este modelo puede dar ni si una versión menos compleja sería más estable.

**Qué tan seguro estoy.** Confirmado por el código.

### H3 — El umbral de disparidad está escrito, pero no justificado por el costo del proyecto (D3)

**Qué encontré.** `optimizacion.py` línea 46: `UMBRAL_DISPARIDAD, MIN_MALOS = 0.10, 30`; líneas 21-23: `"Umbral de disparidad (hallazgo D3): caída de 0.10 absoluto en malos detectados de un subgrupo frente al global (valor por omisión de la Skill). Un subgrupo con menos de 30 malos se marca como no concluyente (hallazgo D4)."` La única justificación es que es el valor por omisión.

**Por qué importa.** La Skill presenta 0.10 como «un valor de arranque, no una constante universal» y pide que «cada proyecto [lo reemplace] por un umbral argumentado según el costo de su error». Aquí aprobar un malo cuesta 5 veces más. Una caída de 0.10 en recall dentro de un grupo de 186 malos son unos 19 malos aprobados más, es decir 93 unidades de costo; nadie dice si eso es tolerable. Con la medición actual no cambia ningún resultado: ningún subgrupo concluyente se acerca a 0.10.

**Qué tan seguro estoy.** Confirmado por el código. Que la justificación por omisión no baste es la lectura literal de la Skill; si el lector acepta el valor por omisión como decisión consciente y declarada, D3 sería el único veredicto que cambiaría.

### H4 — La confirmación de la regla de elección no cubre a todos los rivales elegibles (E4)

**Qué encontré.** La regla (líneas 18-19) exige que el elegido tenga «el menor costo también en las 5 semillas adicionales». La salida que debería probarlo, `robustez_semillas_cv` (`optimizacion.json` desde la línea 216; candidatos en `optimizacion.py` líneas 128-133), compara al elegido con tres rivales y omite dos que cumplen el filtro de exactitud: Árbol afinado (anidada), con 0.714, y Random Forest balanceado, con 0.757. En segundo plano, la docstring dice `"ningún modelo usa VARIABLES_SENSIBLES"` (línea 14), mientras la curva de diagnóstico de la 6.1 (línea 144) entrena un árbol con todas las columnas. Está rotulado `# tal como en la 6.1` y no interviene en la elección, pero la frase es más amplia que el código.

*Medición propia de la auditoría*: con las semillas 0-4, Random Forest balanceado cuesta 990, 956, 975, 1039 y 976, y Árbol afinado (anidada) 1014, 1002, 1053, 1059 y 1003, frente a 736, 727, 738, 760 y 755 del elegido (`optimizacion.json`, `robustez_semillas_cv`). La afirmación es cierta en sustancia.

**Por qué importa.** Es el paso que el proyecto añadió para responder al hallazgo A6, y una confirmación que omite candidatos no confirma la regla tal como está escrita. El efecto sobre la decisión es nulo según la medición propia; el defecto es documental.

**Qué tan seguro estoy.** Confirmado por el código (candidatos omitidos). La sustancia, por medición propia.

**Lo que sí quedó corregido de E4.** *Medición propia de la auditoría*: el pipeline del elegido ajustado tiene `n_features_in_ = 17` y `get_feature_names_out()` no contiene `age`, `personal_status` ni `foreign_worker`. Al permutar al azar esas tres columnas, las probabilidades predichas son idénticas. La afirmación «se usan para auditar, no para decidir» (`modelos.py` línea 9; `optimizacion.py` líneas 14-15 y 160) es verdadera como uso directo.

## Observaciones nuevas (no cambian veredictos)

- **`housing` sigue siendo predictora.** `modelos.py` líneas 10-11: `"'housing' no es un atributo protegido: se queda como variable financiera, pero también se audita."` El dueño la había declarado columna de auditoría, y el proyecto ahora decide de forma explícita lo contrario. No es una afirmación falsa, pero contradice la entrada de esta auditoría y conviene que el dueño la confirme.
- **La exclusión no impide el uso indirecto.** *Medición propia de la auditoría*: con las 17 predictoras que usa el modelo, un boosting predice los atributos excluidos fuera de pliegue con AUC 0.807 (`foreign_worker = yes`), 0.767 (edad ≤ 24) y 0.692 (sexo femenino según `personal_status`). Retirar las columnas elimina el uso directo, no la información.
- **Disparidades estables bajo el umbral.** *Medición propia de la auditoría* con las semillas de CV 0-4: `housing = own` cae entre 0.038 y 0.089 en las cinco (máximo 0.089), y `rent` y `for free` quedan siempre por encima del global. La mayor caída puntual de un grupo concluyente es 0.090 (edad 50+, 34 malos, semilla 0). Los grupos no concluyentes oscilan mucho: `foreign_worker = no` va de −0.43 a 0.09 y `male mar/wid` de −0.09 a 0.077.
- **Métrica secundaria.** La mayor caída de exactitud es 0.092 (`male div/sep`, 50 filas), bajo 0.10.
- **Lectura de `supera_umbral`.** En los grupos no concluyentes la columna vale `False` (`foreign_worker = no`, caída 0.10; `male mar/wid`, caída 0.12). La columna `concluyente = False` aclara que no están aprobados, pero un `False` en `supera_umbral` se puede leer como aprobado.
- **Umbral de decisión.** *Medición propia de la auditoría*: con probabilidad ≥ 1/6 (el umbral que implica la matriz 5:1), el elegido da recall 0.7633 y costo 648, frente a 0.60 y 728 con `predict()`; la exactitud baja a 0.636 y quedaría fuera del filtro > 0.70 de la regla.

## Estado de los hallazgos de la primera ronda

| Primera ronda | Estado | Motivo |
|---|---|---|
| E4 — variables sensibles como predictoras | Corregido | Excluidas del pipeline (`modelos.py` líneas 12, 19-21); comprobado con nombres de salida y permutación |
| E4 — elección sin regla escrita | Corregido en parte | Regla escrita (`optimizacion.py` líneas 17-19), pero su confirmación omite dos rivales elegibles → E4 sigue `FALLA` (H4) |
| A6 — selección y evaluación con la misma partición | Abierto | La regla escrita no cambia el procedimiento; no hay evaluación anidada ni prueba reservada (H1). Efecto medido: nulo |
| E3 — capacidad no explorada | Abierto | Sin cambios en `modelos.py` línea 42 (H2) |
| D3 — sin umbral | Corregido en parte | Escrito (línea 46), justificado solo como valor por omisión (H3) |
| D4 — grupos pequeños sin rótulo | Corregido | Columnas `malos_en_grupo` y `concluyente` (líneas 168-171) |

## Acciones recomendadas

1. `optimizacion.py` sección 3 — Estimar el desempeño del elegido aplicando la regla de elección dentro de cada pliegue externo (selección anidada), o con una prueba reservada que no participe en la elección, y reportar esas cifras como las del modelo recomendado. — Es lo único que convierte A6 en `PASA`; la medición propia sugiere que las cifras no cambiarán.
2. `modelos.py` línea 42 / `busqueda.py` — Explorar `max_iter`, `max_leaf_nodes` o `max_depth`, `learning_rate` y `min_samples_leaf` del boosting con validación anidada, reportando juntos entrenamiento y validación (exactitud y recall de «bad») y puntuando por costo. — La brecha 0.996 frente a 0.752 sigue sin respuesta.
3. `optimizacion.py` líneas 21-23 — Argumentar el umbral de disparidad en términos del costo del proyecto (p. ej. cuántos malos aprobados adicionales o cuántas unidades de costo por grupo se toleran), o declarar por qué el valor por omisión es aceptable para este dominio. — D3.
4. `optimizacion.py` líneas 128-133 — Incluir en `robustez` a todos los candidatos que pasan el filtro de exactitud (Árbol afinado anidado y Random Forest balanceado); ajustar la línea 14 para que diga «ningún modelo comparado» o equivalente. — E4.
5. `modelos.py` líneas 10-11 — Confirmar con el dueño si `housing` debe ser predictora; documentar en el informe que las predictoras restantes permiten reconstruir en parte los atributos excluidos (AUC 0.69-0.81). — Exclusión directa no es ceguera.
6. `optimizacion.py` línea 171 — Dejar `supera_umbral` vacío (o «no concluyente») en los grupos con menos de 30 malos, en lugar de `False`. — Evita leerlos como aprobados.

## Lo que esta auditoría no revisó

- **Archivos excluidos:** `tests/`, `.claude/` (salvo la Skill), `entregas/`, `evidencia/`, `auditoria/` y `experimentos.py`; tampoco `main.py`, que no estaba en la lista y sigue llamando a `crear_pipeline` sin `excluir`, es decir, con las variables sensibles.
- **Informe de la Tarea 6.2:** `modelos.py` línea 41 aún remite a «el informe», que no está en el repositorio.
- **Selección anidada completa:** la medición propia de A6 dejó fuera a los dos árboles con búsqueda anidada por su costo de cómputo.
- **Entidades repetidas (B6):** sin identificador de solicitante; solo se descartaron duplicados exactos.
- **C2** depende de criterio de dominio (momento de captura de cada campo).
- **Equidad más allá del recall:** no se midieron tasas de rechazo de buenos por subgrupo, calibración por grupo ni subgrupos interseccionales.
- **Representatividad, calidad de las etiquetas y vigencia de la matriz de costos** de Hofmann (1994).
- **Las mediciones propias** usan particiones de los mismos 1,000 casos; no hay datos externos.
