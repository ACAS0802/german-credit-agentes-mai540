# Auditoría — german-credit-agentes-mai540 (Tarea 6.2, modelo recomendado)

**Archivos auditados:** `datos.py` (20 líneas), `modelos.py` (32), `validacion.py` (28), `busqueda.py` (24), `curvas.py` (25), `auditoria.py` (17), `optimizacion.py` (187); salidas de `optimizacion.py`: `resultados/optimizacion.json` (477 líneas), `resultados/busqueda_arbol.csv` (41), `resultados/subgrupos_edad_grupo.csv` (5), `resultados/subgrupos_personal_status.csv` (5), `resultados/subgrupos_housing.csv` (4), `resultados/subgrupos_foreign_worker.csv` (3), `figs/curvas_6_1_vs_6_2.png`. Excluidos por indicación del dueño: `tests/`, `.claude/`, `entregas/`, `evidencia/`. `experimentos.py` (148 líneas) se excluyó por irrelevante: solo evalúa el árbol y el Random Forest de la 6.1 y escribe en `entregas/`; no toca el modelo auditado.
**Modelo auditado:** `modelos.modelo_elegido()` — `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)` dentro de `crear_pipeline`, evaluado con la validación cruzada externa de `datos.py` (`StratifiedKFold(5, shuffle=True, random_state=42)`).
**Tipo de problema:** clasificación binaria (declarado por el dueño; coincide con el código: `datos.py` línea 9 crea un objetivo 0/1 y los estimadores son `*Classifier`).
**Objetivo:** `clase` (y = 1 si `"bad"`)   **Clase de interés:** `"bad"` = 1 (aprobar un crédito malo cuesta 5 veces más que rechazar uno bueno)
**Columnas de subgrupo:** edad agrupada (19-24, 25-34, 35-49, 50+), `personal_status`, `housing`, `foreign_worker` — declaradas como columnas solo de auditoría.
**Métrica prioritaria:** recall de `"bad"` (en el proyecto, `malos_detectados`); secundaria: exactitud.
**Umbral de disparidad:** 0.10 absoluto sobre el recall de `"bad"` (valor por omisión de la Skill; el proyecto no fija uno propio).
**Convención de citas:** no hay notebooks; las citas son `archivo línea N`, con líneas numeradas desde 1 tal como las muestra `cat -n`. Las mediciones propias se rotulan *medición propia de la auditoría*; sus scripts se ejecutaron fuera del repositorio (`/tmp/claude-0/skill62/`) y no escriben nada dentro del proyecto.
**Fecha:** 2026-10-06

## Resumen

De 25 verificaciones, 19 pasan, 5 fallan (A6, D3, D4, E3, E4) y 1 no aplica (C4); ninguna quedó en `NO SE PUEDE DETERMINAR`. No hay fuga ni errores de cálculo: las cifras del modelo elegido (exactitud 0.746 ± 0.021, recall de «bad» 0.5467, costo 798) se reproducen exactamente y son coherentes con su matriz de confusión, así que se pueden creer como estimación de validación cruzada. Lo más grave es que una afirmación de diseño es falsa: el proyecto dice que las variables sensibles «solo auditan», pero `age`, `personal_status`, `housing` y `foreign_worker` son predictoras del modelo. Además, el modelo se eligió comparando candidatos sobre las mismas particiones que lo evalúan, sin regla de elección escrita, y su capacidad no se exploró. Medido por la auditoría, ningún subgrupo con al menos 30 malos pierde más de 0.10 de recall (la mayor caída es 0.060, edad 35-49), pero el proyecto no fija un umbral ni marca como no concluyentes los grupos con pocos malos.

## Tabla de verificación

| # | Verificación | Veredicto | Evidencia |
|---|---|---|---|
| A1 | Métricas en rango válido | PASA | `resultados/optimizacion.json` líneas 192-206 (modelo elegido): `"exactitud_media": 0.746`, `"malos_detectados": 0.5467`; todas las exactitudes y recalls del JSON están entre 0.3367 y 0.79 |
| A2 | Métricas consistentes con la matriz de confusión | PASA | `optimizacion.json` líneas 202-205: `"malos_aprobados": 136, "buenos_rechazados": 118, "costo": 798`. Recalculado: recall = (300−136)/300 = 0.5467; exactitud = 1 − 254/1000 = 0.746; costo = 5·136 + 118 = 798. Las otras seis filas de `comparacion` y `busqueda` cuadran igual (p. ej. árbol por defecto: 159/300 = 0.53, 1 − 302/1000 = 0.698, 5·141 + 161 = 866) |
| A3 | Con clases desbalanceadas se reporta más que la exactitud | PASA | 30 % de «bad» (`datos.py` línea 19: `# 0.3: el 30% es malo`); `optimizacion.py` líneas 50-52 reportan `"malos_detectados"`, `"malos_aprobados"`, `"buenos_rechazados"` y `"costo"` |
| A4 | La métrica prioritaria corresponde al costo del error y está argumentada | PASA | `optimizacion.py` líneas 11-12: `"Costo: 5 × (malos aprobados) + 1 × (buenos rechazados), la matriz de costos del dataset original (Hofmann, 1994). Aprobar un crédito malo cuesta 5 veces más."`; `modelos.py` líneas 30-31: `"Aprobar un crédito malo cuesta más que rechazar uno bueno; con class_weight='balanced' cada error sobre un «bad» pesa más al entrenar"` |
| A5 | `pos_label` apunta a la clase de interés en toda métrica binaria | PASA | `datos.py` línea 9: `y = (df["clase"] == "bad").astype(int)`; `validacion.py` línea 17: `recall_score(y, pred, pos_label=1)`; `optimizacion.py` línea 50: `recall_score(y, pred)` (por omisión `pos_label=1` = «bad»); línea 45: `tn, fp, fn, tp = confusion_matrix(y, pred).ravel()` (orden correcto para etiquetas 0/1); `auditoria.py` línea 11: `malos = tabla[tabla["real"] == 1]` |
| A6 | Ningún hiperparámetro se eligió con la partición que produce la métrica reportada | FALLA | `optimizacion.py` líneas 97-109: los seis candidatos (incluidos `class_weight=None` frente a `"balanced"`) se comparan con `evaluar(..., X, y, cv)` sobre la misma `cv` de la que salen las cifras del elegido (línea 109: `comparacion["Boosting balanceado (elegido)"] = evaluar(crear_pipeline(modelo_elegido(), cols_cat), X, y, cv)`); no hay prueba reservada ni selección anidada para el modelo elegido. Ver H2 |
| B1 | La partición ocurre antes de toda transformación que aprende | PASA | `modelos.py` líneas 11-16: el `OrdinalEncoder` está dentro de `Pipeline([("prep", prep), ("modelo", modelo)])`; `optimizacion.py` líneas 43-44: `cross_validate(estimador, X, y, cv=cv, ...)` y `cross_val_predict(estimador, X, y, cv=cv, ...)` ajustan el pipeline completo dentro de cada pliegue |
| B2 | Semilla fija en partición y estimadores | PASA | `datos.py` línea 12: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`; `modelos.py` línea 32: `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)`; `optimizacion.py` línea 36: `CV_INTERNA = StratifiedKFold(..., random_state=1)`; `busqueda.py` línea 21: `random_state=42`; `curvas.py` línea 14: `shuffle=True, random_state=42` |
| B3 | Partición estratificada | PASA | `datos.py` línea 12: `StratifiedKFold(...)`; `busqueda.py` línea 21: `stratify=y` |
| B4 | Los modelos comparados usan la misma partición y datos | PASA | `optimizacion.py` líneas 98-109: todos los candidatos pasan por `evaluar(..., X, y, cv)` con los mismos `X`, `y`, `cv` obtenidos en la línea 58 (`X, y, cols_cat, cv = cargar_datos()`) |
| B5 | La validación cruzada recibe un estimador sin entrenar | PASA | `optimizacion.py` línea 109: `evaluar(crear_pipeline(modelo_elegido(), cols_cat), X, y, cv)` construye un pipeline nuevo; `evaluar` (líneas 43-44) solo llama a `cross_validate` / `cross_val_predict`, que clonan. El único `.fit` previo sobre un objeto reutilizado (línea 81, `gs_anidado.fit(...)`) ocurre después de evaluarlo (línea 80) |
| B6 | Ninguna fila en entrenamiento y prueba a la vez | PASA | *Medición propia de la auditoría* sobre `datos/credit_g.csv`: 0 filas duplicadas y 0 vectores de predictoras duplicados entre las 1,000 filas. El conjunto no tiene identificador de solicitante (ver *Lo que esta auditoría no revisó*) |
| C1 | Ningún transformador se ajusta con datos de prueba | PASA | `modelos.py` línea 16: `return Pipeline([("prep", prep), ("modelo", modelo)])  # 'prep' se ajusta solo con los datos de entrenamiento de cada pliegue`; único transformador del proyecto |
| C2 | Ninguna predictora es posterior a la predicción | PASA | `datos.py` línea 10: `X = df.drop(columns="clase")`. Las 20 predictoras (`checking_status`, `duration`, `credit_history`, `purpose`, `credit_amount`, `savings_status`, `employment`, `installment_commitment`, `personal_status`, `other_parties`, `residence_since`, `property_magnitude`, `age`, `other_payment_plans`, `housing`, `existing_credits`, `job`, `num_dependents`, `own_telephone`, `foreign_worker`) son datos de la solicitud; ningún nombre del tipo `*_final`, `*_programad*`, `resultado_*` |
| C3 | Ninguna predictora separa el objetivo completo | PASA | *Medición propia de la auditoría*: tabla de cada predictora contra `y`. Disparó la comprobación `credit_amount` (94.2 % de las filas caen en valores «puros»), pero es un artefacto de cardinalidad (921 valores distintos en 1,000 filas): un árbol de una sola variable con validación cruzada logra 0.572, por debajo del 0.70 de «siempre bueno». Descartada. Ninguna otra predictora pasa de 3.1 % de filas en valores puros |
| C4 | Selección de características ajustada solo con entrenamiento | NO APLICA | El proyecto no selecciona características: no hay `SelectKBest`, `mutual_info` ni filtros por correlación en los siete archivos; `datos.py` línea 10 entrega las 20 columnas al pipeline |
| C5 | El objetivo no está entre las predictoras | PASA | `datos.py` líneas 9-10: `y = (df["clase"] == "bad").astype(int)` / `X = df.drop(columns="clase")`; ninguna columna de `X` deriva de `clase` |
| D1 | Métrica prioritaria calculada por subgrupo | PASA | `auditoria.py` línea 16: `"malos_detectados": malos.groupby("grupo")["acierto"].mean(),   # recall de «bad» dentro del grupo`; `optimizacion.py` líneas 149-153 lo aplican a las cuatro columnas de subgrupo y escriben `resultados/subgrupos_*.csv` |
| D2 | Ningún subgrupo con ≥ 30 casos cae más de 0.10 bajo el global | PASA | *Medición propia de la auditoría* (recall global 0.5467). Con el denominador correcto para recall (casos «bad» del subgrupo), los subgrupos con ≥ 30 malos tienen como mayor caída 0.0602 (edad 35-49, 74 malos) y 0.0574 (`housing = own`, 186 malos). Los valores del proyecto coinciden: `resultados/subgrupos_edad_grupo.csv` línea 4: `35-49,327,0.8043,0.2263,0.4865` |
| D3 | El umbral está escrito y justificado en el proyecto | FALLA | `optimizacion.py` líneas 145-154 calculan y guardan las tablas (`tabla.round(4).to_csv(f"resultados/subgrupos_{col}.csv")`) sin compararlas con ningún umbral; en los siete archivos auditados no aparece un umbral de disparidad (la única aparición de «umbral» está en `experimentos.py` y se refiere al umbral de decisión). Ver H4 |
| D4 | Subgrupos con < 30 casos reportados como no concluyentes | FALLA | `resultados/subgrupos_foreign_worker.csv` línea 2: `no,37,0.8649,0.1081,0.25` (recall sobre 4 malos); `resultados/subgrupos_personal_status.csv` línea 4: `male mar/wid,92,0.6739,0.2717,0.4` (25 malos) y línea 3: `male div/sep,50,0.68,0.4,0.6` (20 malos). Se publican junto a los demás sin rótulo; `auditoria.py` líneas 12-17 no reportan el número de malos ni un tamaño mínimo. Ver H4 |
| E1 | Sin valores no finitos ni filas descartadas | PASA | *Medición propia de la auditoría*: 0 `NaN` y 0 `inf` en `datos/credit_g.csv`; `cross_val_predict` devuelve 1,000 predicciones para 1,000 filas (`optimizacion.json`: 582 + 118 + 136 + 164 = 1,000) |
| E2 | Reproducibilidad y consistencia interna | PASA | *Medición propia de la auditoría*: reejecutar el protocolo de `optimizacion.py` líneas 43-53 sobre el elegido da pliegues `[0.745, 0.735, 0.725, 0.78, 0.745]`, desviación 0.0207, `tn fp fn tp = 582 118 136 164`, idénticos a `optimizacion.json` líneas 194-205; dos ejecuciones de `cross_val_predict` dan predicciones idénticas. `"optimismo_busqueda": 0.031` (línea 108) = 0.73 − 0.699 |
| E3 | Capacidad elegida con entrenamiento y validación reportados | FALLA | `modelos.py` línea 32: `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)` — `max_iter` (número de iteraciones de boosting), `max_leaf_nodes`, `max_depth`, `learning_rate` y `min_samples_leaf` quedan en sus valores por omisión (100, 31, None, 0.1, 20) sin ninguna exploración; `PARAM_GRID` (`busqueda.py` líneas 5-8) solo se aplica al árbol. Ver H3 |
| E4 | Afirmaciones respaldadas por salidas, salidas interpretadas | FALLA | `optimizacion.py` línea 145: `# 5. Auditoría por subgrupo del modelo elegido (las variables sensibles solo auditan)` — contradicho por `datos.py` línea 10 (`X = df.drop(columns="clase")`), que deja `age`, `personal_status`, `housing` y `foreign_worker` como predictoras. `modelos.py` líneas 29-31 remiten a «el informe» para justificar la recomendación, y ese informe no está entre los archivos auditados. Ver H1 y H2 |

## Hallazgos

### H1 — Las variables sensibles son predictoras del modelo, no solo columnas de auditoría (E4)

**Qué encontré.** El proyecto afirma en `optimizacion.py` línea 145:

```python
    # 5. Auditoría por subgrupo del modelo elegido (las variables sensibles solo auditan)
```

y en la docstring de `_por_grupo_externo` (líneas 162-163): `"para un grupo que no es columna del modelo (la edad agrupada): el modelo se entrena con X tal cual; el grupo solo se usa para leer los resultados."` Pero `X` es todo el CSV menos el objetivo (`datos.py` línea 10: `X = df.drop(columns="clase")`), y ese `X` es el que recibe el modelo elegido (`optimizacion.py` línea 146: `elegido = crear_pipeline(modelo_elegido(), cols_cat)`, evaluado en las líneas 150-151 con `X`). Las cuatro columnas de subgrupo son, por tanto, entradas del modelo: `age` como numérica en bruto y `personal_status`, `housing`, `foreign_worker` codificadas con `OrdinalEncoder` (`modelos.py` líneas 11-13). La edad *agrupada* no es columna del modelo, pero la edad sí. `personal_status` mezcla sexo y estado civil (`female div/dep/mar`, `male single`, …).

*Medición propia de la auditoría* (importancia por permutación fuera de pliegue, exactitud balanceada, 10 repeticiones): `age` es la 8.ª de 20 predictoras (0.0049), `personal_status` la 11.ª (0.0033), `housing` la 14.ª (0.0010) y `foreign_worker` la 17.ª (−0.0023). Al quitar las cuatro columnas y repetir el mismo protocolo con la `cv` de `datos.py`, el modelo da exactitud 0.749, recall de «bad» 0.59 y costo 743, frente a 0.746 / 0.5467 / 798 con ellas; con las semillas de CV 0-4 el costo sin ellas es 762, 760, 706, 772, 720 frente a 748, 760, 750, 768, 744 con ellas.

**Por qué importa.** No infla las métricas, pero cambia qué describen: 0.746 y 0.5467 son el desempeño de un modelo que decide también con edad, sexo/estado civil y condición de trabajador extranjero, no de uno que solo las usa para auditar. La auditoría por subgrupo del proyecto mide así la disparidad de un modelo que tiene acceso directo a esos atributos, y la afirmación escrita induce a error a quien lea el código o el informe. La medición propia sugiere que retirarlas no cuesta desempeño (es similar o mejor en 6 de 6 particiones probadas), así que la contradicción no se sostiene por necesidad técnica.

**Qué tan seguro estoy.** Confirmado por el código (que las columnas entran al modelo). La comparación con y sin ellas es una medición propia sobre seis particiones de los mismos 1,000 casos; no se probó con datos nuevos.

### H2 — El modelo se eligió con las mismas particiones que producen sus cifras, y la elección no tiene regla escrita (A6, E4)

**Qué encontré.** `optimizacion.py` líneas 97-109 evalúan seis candidatos con `evaluar(..., X, y, cv)` y la línea 109 rotula uno como elegido:

```python
    comparacion["Boosting balanceado (elegido)"] = evaluar(crear_pipeline(modelo_elegido(), cols_cat), X, y, cv)
```

Las cifras que se reportan del elegido (0.746, 0.5467, 798) salen de esa misma `cv`; la decisión entre `class_weight=None` y `"balanced"` y entre familias de modelos se tomó mirando esas mismas particiones (y las semillas 0-4 de las líneas 120-123, que reparten los mismos 1,000 casos). No hay prueba reservada ni selección anidada para el elegido; el árbol sí la tiene (líneas 79-80). Ninguno de los siete archivos dice qué criterio decidió la elección: `modelos.py` líneas 29-31 remiten a `"ver optimizacion.py y el informe"`, y `optimizacion.py` no contiene ese criterio. Con la `cv` principal, la propia salida del proyecto muestra otro candidato mejor en la métrica prioritaria y en costo: `optimizacion.json` líneas 160-174, `"Árbol afinado balanceado (anidada)"`: `"malos_detectados": 0.6533`, `"costo": 719`, frente a 0.5467 y 798 del elegido. La elección solo se entiende con `robustez_semillas_cv` (líneas 209-319), donde el elegido tiene el menor costo en las cinco semillas (744-768 frente a 811-896 del árbol balanceado), pero ningún texto lo interpreta.

*Medición propia de la auditoría*: elegir `class_weight` ∈ {None, "balanced"} por costo con una CV interna (`random_state=1`) dentro de cada pliegue externo elige `"balanced"` en los 5 pliegues y reproduce exactamente 0.746 / 0.5467 / 798. Para esa decisión, el optimismo medido es 0.

**Por qué importa.** Un número usado para elegir entre varios candidatos sobrestima, en promedio, el desempeño del ganador. Aquí el efecto medido de `class_weight` es nulo, pero la elección de familia (boosting frente a árbol balanceado frente a RF) no está anidada, y sin una regla escrita no se puede saber si las cifras del elegido son la mejor de varias miradas. Además, quien lea solo la tabla `comparacion` vería que el modelo recomendado no es el mejor en la métrica que el proyecto declara prioritaria.

**Qué tan seguro estoy.** Confirmado que la selección y la evaluación comparten partición (código). Que el sesgo sea pequeño está medido solo para `class_weight`; para la elección de familia es inferencia. Faltaría una evaluación anidada de la regla de elección completa, o una prueba reservada que no se haya usado para elegir.

### H3 — La capacidad del modelo elegido no se exploró (E3)

**Qué encontré.** `modelos.py` línea 32: `return HistGradientBoostingClassifier(class_weight="balanced", random_state=42)`. El modelo tiene hiperparámetros de capacidad: `max_iter` (en este estimador es el número de árboles del boosting, no la convergencia de un optimizador), `max_leaf_nodes` = 31, `max_depth` = None, `learning_rate` = 0.1, `min_samples_leaf` = 20, todos por omisión. La única búsqueda (`busqueda.py` líneas 5-8, `PARAM_GRID` con `modelo__max_depth` y `modelo__min_samples_leaf`) se aplica al árbol (`optimizacion.py` líneas 68-69, 79, 103-104). La curva de `curvas.py` varía el tamaño de entrenamiento, no la capacidad; su salida muestra el síntoma: `optimizacion.json` líneas 366-386, entrenamiento 0.9992 frente a validación 0.746, `"brecha_final": 0.2532`.

**Por qué importa.** Con 800 casos de entrenamiento el modelo casi memoriza (0.999). Sin explorar la capacidad no se sabe si un modelo menos complejo daría el mismo recall con menos varianza, ni si 0.746 / 0.5467 es lo que este tipo de modelo puede dar o un punto arbitrario. La brecha es un síntoma; la Skill pide la evidencia de cómo se eligió la capacidad, y no existe.

**Qué tan seguro estoy.** Confirmado por el código: no hay exploración de los hiperparámetros del boosting en ninguno de los siete archivos.

### H4 — La auditoría por subgrupo no tiene regla de decisión: ni umbral ni tamaño mínimo (D3, D4)

**Qué encontré.** `optimizacion.py` líneas 149-153 calculan y guardan el recall por subgrupo, pero ningún archivo fija un umbral de disparidad ni un número mínimo de casos. `auditoria.py` líneas 12-17 devuelven `casos`, `exactitud`, `prop_malos` y `malos_detectados`, pero no el número de malos, que es el denominador del recall. Así se publican al mismo nivel que los demás:

- `resultados/subgrupos_foreign_worker.csv`: `no,37,0.8649,0.1081,0.25` — recall 0.25 calculado sobre **4** malos (1 de 4).
- `resultados/subgrupos_personal_status.csv`: `male mar/wid,92,0.6739,0.2717,0.4` — recall 0.40 sobre **25** malos; y `male div/sep,50,0.68,0.4,0.6` — sobre **20** malos.

*Medición propia de la auditoría* (recall global 0.5467; caída = global − subgrupo; Fisher exacta del subgrupo contra el resto, entre los «bad»):

| Columna | Grupo | Filas | Malos | Recall | Caída | p (Fisher) | Exactitud |
|---|---|---|---|---|---|---|---|
| edad | 19-24 | 149 | 61 | 0.5902 | −0.0435 | 0.474 | 0.6510 |
| edad | 25-34 | 399 | 131 | 0.5573 | −0.0106 | 0.815 | 0.7243 |
| edad | 35-49 | 327 | 74 | 0.4865 | 0.0602 | 0.282 | 0.8043 |
| edad | 50+ | 125 | 34 | 0.5588 | −0.0122 | 1.000 | 0.7760 |
| personal_status | female div/dep/mar | 310 | 109 | 0.5780 | −0.0313 | 0.470 | 0.7290 |
| personal_status | male div/sep | 50 | **20** | 0.6000 | −0.0533 | 0.651 | 0.6800 |
| personal_status | male mar/wid | 92 | **25** | 0.4000 | **0.1467** | 0.144 | 0.6739 |
| personal_status | male single | 548 | 146 | 0.5411 | 0.0056 | 0.908 | 0.7737 |
| housing | for free | 108 | 44 | 0.6136 | −0.0670 | 0.413 | 0.6944 |
| housing | own | 713 | 186 | 0.4892 | 0.0574 | 0.012 | 0.7574 |
| housing | rent | 179 | 70 | 0.6571 | −0.1105 | 0.040 | 0.7318 |
| foreign_worker | no | 37 | **4** | 0.2500 | **0.2967** | 0.332 | 0.8649 |
| foreign_worker | yes | 963 | 296 | 0.5507 | −0.0040 | 0.332 | 0.7414 |

Las dos caídas mayores de 0.10 (`male mar/wid` 0.1467 y `foreign_worker = no` 0.2967) ocurren en grupos con menos de 30 malos y no se confirman con la prueba de proporciones (p = 0.144 y 0.332). Con las semillas de CV 0-4, la caída de `male mar/wid` va de −0.017 a 0.093 y la de `foreign_worker = no` de −0.187 a 0.333: son inestables, como se espera con 25 y 4 observaciones. Entre los grupos con al menos 30 malos, ninguno cae más de 0.10 (máximo 0.0602, edad 35-49; con las semillas 0-4, máximo 0.0698), de ahí el `PASA` de D2. Hay una diferencia sistemática por debajo del umbral: `housing = own` tiene recall 0.4892 (caída 0.057, p = 0.012) y su caída es positiva en las seis particiones (0.042-0.065), mientras `rent` está siempre por encima del global. En la métrica secundaria, la mayor caída de exactitud es 0.095 (edad 19-24, 149 filas), justo debajo de 0.10.

**Por qué importa.** Sin umbral, la tabla del proyecto no dice si el modelo pasa o no su propia auditoría de equidad; sin denominador ni tamaño mínimo, un lector puede tomar el 0.25 de `foreign_worker = no` como una disparidad grave (es 1 de 4) o el 0.86 de exactitud de ese grupo como un buen resultado, cuando ninguno de los dos es concluyente.

**Qué tan seguro estoy.** Confirmado por el código y las salidas (falta de umbral y de rótulo). Las cifras por subgrupo son medición propia, idénticas a las del proyecto donde se solapan.

**Nota de interpretación (D2).** Se aplicó la regla de la Skill sobre el denominador: con el recall como métrica prioritaria, el umbral de 30 se cuenta en casos «bad» del subgrupo, no en filas. Leído en filas, `male mar/wid` (92 filas) quedaría dentro de D2 con una caída de 0.1467 y D2 sería `FALLA`. Se dejó en D4 porque su recall descansa en 25 observaciones y la diferencia no se confirma estadísticamente.

## Acciones recomendadas

1. `datos.py` línea 10 / `optimizacion.py` líneas 145 y 161-163 — Decidir explícitamente si `age`, `personal_status`, `housing` y `foreign_worker` entran al modelo. Si el principio es «solo auditan», retirarlas de `X` antes de crear el pipeline y conservarlas aparte para el desglose; si se mantienen, corregir el comentario y la docstring y justificarlo por escrito. — La afirmación actual es falsa, y la medición propia indica que quitarlas no empeora el recall ni el costo.
2. `optimizacion.py` sección 3 (líneas 95-124) y `modelos.py` líneas 28-31 — Escribir la regla de elección del modelo (p. ej. «menor costo medio en las semillas de CV 0-4») e interpretar por qué el árbol balanceado, que gana en la `cv` principal (0.6533 / 719), no se eligió. Estimar el desempeño del elegido con la regla de elección anidada, o con una prueba reservada que no se use para elegir. — Sin eso, las cifras del elegido son las de un ganador entre seis vistos en las mismas particiones.
3. `modelos.py` línea 32 / `busqueda.py` — Explorar la capacidad del boosting (`max_iter`, `max_leaf_nodes` o `max_depth`, `learning_rate`, `min_samples_leaf`) con validación anidada, reportando exactitud y recall de «bad» de entrenamiento y de validación juntos, y puntuando por costo o recall en lugar de `scoring="accuracy"`. — La brecha 0.999 frente a 0.746 indica sobreajuste y no hay evidencia de que la configuración por omisión sea adecuada.
4. `optimizacion.py` sección 5 (líneas 145-154) y `auditoria.py` — Fijar y argumentar un umbral de disparidad sobre el recall de «bad» (o adoptar 0.10 diciéndolo), añadir a las tablas el número de malos por grupo, y rotular como «no concluyente» todo grupo con menos de 30 malos (`foreign_worker = no`, `male mar/wid`, `male div/sep`). — Hoy la tabla no dice si el modelo pasa su propia auditoría.
5. Informe de la Tarea 6.2 — Comentar la diferencia sistemática de `housing = own` (recall 0.489, p = 0.012, caída positiva en las seis particiones) y la exactitud de 19-24 (0.651, a 0.095 del global). — Son las señales más estables de la auditoría, aunque estén bajo el umbral.
6. `optimizacion.py` / `modelos.py` (opcional, fuera de las verificaciones) — Evaluar fijar el umbral de decisión por la matriz de costos (probabilidad ≥ 1/6) en lugar de `predict()` (0.5). *Medición propia*: con la `cv` de `datos.py`, el umbral 1/6 da recall 0.7667 y costo 629, frente a 0.5467 y 798; el umbral teórico no se ajustó a los datos. — `class_weight="balanced"` repondera aproximadamente 7:3, no 5:1.

## Lo que esta auditoría no revisó

- **Archivos excluidos por indicación del dueño:** `tests/`, `.claude/` (salvo la propia Skill), `entregas/`, `evidencia/`. No se leyeron.
- **`experimentos.py`** se excluyó por irrelevante para el modelo auditado (solo árbol y Random Forest de la 6.1). Tampoco se auditó `main.py`, que no estaba en la lista y, según el README, no corre completo.
- **Informe de la Tarea 6.2:** `modelos.py` línea 31 remite a «el informe»; en el repositorio solo hay el informe de la 6.1 (`MAI540_Tarea6.1_Informe_Araceli_Castillo.pdf/.docx`), que no se leyó. Si existe un texto de la 6.2 con la regla de elección o el umbral de disparidad, podría cambiar E4, A6 (solo en lo documental) y D3.
- **Entidades repetidas (B6):** el CSV no trae identificador de solicitante; no hay duplicados exactos, pero no se puede descartar que un mismo cliente aparezca con dos solicitudes distintas.
- **C2** depende de criterio de dominio: se juzgó con los nombres de columna y la descripción pública de German Credit (datos de la solicitud), sin documentación del momento de captura de cada campo.
- **Representatividad y etiquetas:** no se auditó cómo se recogieron los 1,000 casos ni la calidad de la etiqueta `clase`, ni la validez actual de la matriz de costos de Hofmann (1994).
- **Subgrupos interseccionales** (p. ej. edad × `personal_status`) y otras variables posiblemente sensibles (`job`, `num_dependents`, `own_telephone`) no se desglosaron.
- **Las mediciones propias** (ablación sin variables sensibles, umbral 1/6, selección anidada de `class_weight`, estabilidad por semilla) se hicieron sobre particiones de los mismos 1,000 casos; no hay datos externos para confirmarlas.
- **La figura** `figs/curvas_6_1_vs_6_2.png` se revisó solo como respaldo de E3/E4; su contenido coincide con `optimizacion.json` líneas 321-387.
