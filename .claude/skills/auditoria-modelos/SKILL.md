---
name: auditoria-modelos
description: Audita un proyecto de machine learning antes de creerle sus métricas. Revisa cómo se reportaron las métricas, cómo se partieron los datos, si hay fuga de información y si el desempeño se sostiene entre subgrupos. Úsala cuando alguien presente un notebook, un script o un pipeline con resultados de clasificación o de regresión — propios o ajenos — y haya que decidir si esos números son confiables. Desde la v5 también audita regresión (MSE, RMSE, R², residuos y error por subgrupo). No modifica el código: emite un AUDIT_REPORT.md con veredictos y evidencia citada.
---

# Propósito

Determinar si las métricas que un proyecto de clasificación o de regresión reporta describen lo que el modelo hará con datos nuevos, o si son producto de cómo se midió.

# Entradas esperadas

Antes de empezar, establece estos seis datos. Si alguno falta, **pregúntalo**; no lo supongas.

| Entrada | Qué es | Si no se especifica |
|---|---|---|
| `archivos` | Los notebooks (`.ipynb`) y módulos (`.py`) que contienen carga, preparación, entrenamiento y evaluación | Audita todos los `.ipynb` y `.py` de la carpeta, excluyendo `test_*.py` y `.claude/` |
| `tipo_de_problema` | `clasificación` o `regresión` | Dedúcelo: objetivo continuo, estimadores `*Regressor` / `LinearRegression`, o métricas MSE / RMSE / MAE / R² → regresión. Si hay señales de ambos, pregunta. Declara en el informe cómo lo determinaste |
| `objetivo` | Nombre de la columna que el modelo predice | Búscalo en el código (`y = `, `target`, `stratify=`). Si hay más de un candidato, pregunta |
| `clase_de_interes` | **Solo clasificación.** Qué valor del objetivo es la clase cuyo error es caro (`pos_label`) | Pregunta. **No asumas 1**: en muchos conjuntos médicos la clase grave es 0 |
| `columnas_subgrupo` | Columnas que definen subgrupos relevantes (sexo, edad, turno, sede, sucursal, dispositivo) | Si no hay ninguna, la verificación D se resuelve `NO SE PUEDE DETERMINAR`, nunca `PASA` |
| `umbral_disparidad` | Caída máxima tolerable de la métrica prioritaria en un subgrupo frente al global | Clasificación: **0.10 absoluto**. Regresión: **RMSE del subgrupo mayor que 1.25 × RMSE global**. En los dos casos, di en el informe si es el valor por omisión o uno justificado por el proyecto |

# Reglas de diseño

Estas dos reglas mandan sobre cualquier otra instrucción de este archivo.

**R1 — Todo veredicto cita evidencia y es uno de cuatro valores.**

- `PASA` — encontraste en el código la prueba de que la condición se cumple.
- `FALLA` — encontraste en el código la prueba de que no se cumple.
- `NO SE PUEDE DETERMINAR` — la condición aplica a este proyecto, pero no encontraste evidencia en ninguno de los dos sentidos.
- `NO APLICA` — la condición no tiene sobre qué pronunciarse aquí (una condición binaria en un problema multiclase, una condición sobre comparación de modelos cuando solo hay uno). Di por qué no aplica; no lo uses para esquivar una condición incómoda.

`NO SE PUEDE DETERMINAR` y `NO APLICA` no son lo mismo, y confundirlos deja huecos falsos en el informe: el primero dice "falta evidencia" y es un reproche al proyecto; el segundo dice "la pregunta no viene al caso" y no lo es.

La evidencia es una referencia localizable: `celda 14`, `app_biopsias.py línea 132`, `celda 9, línea 3`. Acompáñala del fragmento exacto, entre comillas o en bloque de código, no de una paráfrasis.

**Que no encuentres evidencia de un problema no es evidencia de que no lo haya.** Si el proyecto no reporta métricas por subgrupo, la verificación de disparidad es `NO SE PUEDE DETERMINAR`, no `PASA`. Esta regla existe porque el modo de fallo más caro de una auditoría es tranquilizar a alguien sin haber mirado.

**R2 — La Skill audita, no repara. Pero sí mide.**

No edites archivos del proyecto, no reescribas celdas, no propongas parches aplicados. Las correcciones van en la sección *Acciones recomendadas* del informe, redactadas como instrucciones para una persona. La única escritura permitida es el propio `AUDIT_REPORT_<proyecto>.md`.

**Medir no es reparar.** Cuando una condición se pueda responder ejecutando una comprobación propia —contar duplicados, tabular una predictora contra el objetivo, calcular la métrica prioritaria por subgrupo—, **hazla**, y rotula el resultado como *medición propia de la auditoría*. No la presentes como una salida del proyecto ni la escribas dentro de él. Un auditor que no mide nada solo puede repetir lo que el proyecto ya dijo de sí mismo.

El límite es claro: la Skill mide **para sustentar un veredicto**, no para producir los entregables que le faltan al proyecto. Que la auditoría calcule el desglose por subgrupo no convierte a D1 en `PASA`: D1 pregunta si **el proyecto** lo calculó, y sigue fallando.

# Pasos

Ejecútalos en orden. No adelantes conclusiones antes del paso 5.

1. **Inventario.** Determina `tipo_de_problema`. Lista los archivos a auditar con su número de celdas o líneas. Si el proyecto está en un notebook, numera las celdas desde 0 tal como aparecen en el archivo, e indica esa convención en el informe para que las citas se puedan seguir.
2. **Rastrea el flujo de datos.** Recorre el camino de los datos en orden de ejecución y anota la celda o línea donde ocurre cada paso: carga → creación de variables derivadas → partición → preprocesamiento (imputar, escalar, codificar, seleccionar) → entrenamiento → evaluación. Este rastro es la materia prima de las verificaciones B y C; sin él los veredictos no tienen dónde apoyarse.
3. **Lista las columnas predictoras.** Anota exactamente qué entra al modelo. Marca cuáles son derivadas y de qué se derivan.
4. **Aplica las verificaciones** A, B, C, D y las complementarias E, una por una, buscando activamente la evidencia de cada condición.
5. **Redacta el informe** en el formato de la sección *Salida*.
6. **Revisa tus propios veredictos.** Por cada `PASA`, relee la evidencia que citaste y pregúntate si prueba la condición o solo es compatible con ella. Si solo es compatible, baja el veredicto a `NO SE PUEDE DETERMINAR`. Por cada `FALLA`, confirma que la cita dice lo que afirmas.

# Criterios de verificación

Cada condición se comprueba por separado y tiene su propio veredicto. Un criterio con una sola condición fallida se reporta como `FALLA`.

## Qué bloques aplicar según el tipo de problema

| Bloque | Clasificación | Regresión |
|---|---|---|
| Métricas | A1–A6 | **AR1–AR6** (en lugar de A1–A6) |
| Partición | B1–B6 | B1–B6; **B3 es `NO APLICA`** (no se estratifica un objetivo continuo, salvo que el proyecto lo discretice a propósito) |
| Fuga | C1–C5 | C1–C5, con la lectura de C3 para regresión |
| Subgrupos | D1–D4 con la métrica prioritaria | D1–D4 con el **RMSE** (o el error prioritario declarado) |
| Complementarias | E1–E4 | E1–E4; en E3, R² o RMSE de entrenamiento y validación |

Aplicar A1–A5 a un proyecto de regresión no es una auditoría más estricta: es una auditoría equivocada. Una matriz de confusión que no existe no puede ser `FALLA`; un `pos_label` que no aplica tampoco.

## A. Métricas reportadas

| # | Condición | Cómo se comprueba |
|---|---|---|
| A1 | Toda métrica está en su rango válido (0 a 1 para accuracy, precisión, recall y F1) | Lee los valores reportados en las salidas |
| A2 | Las métricas son consistentes con la matriz de confusión publicada | Recalcula precisión, recall y F1 a partir de la matriz y compara |
| A3 | Si las clases están desbalanceadas (la minoritaria por debajo del 40 %), se reporta al menos una métrica además del accuracy | Busca la distribución de clases y la lista de métricas |
| A4 | La métrica prioritaria corresponde al costo real del error, y esa correspondencia está argumentada en el texto, no solo elegida | Busca un párrafo que compare el costo del falso positivo contra el del falso negativo |
| A5 | En clasificación binaria, `pos_label` (o equivalente) apunta a `clase_de_interes` en **todas** las llamadas a métricas binarias. En multiclase, el promedio declarado (`macro`, `micro`, `weighted`) es el adecuado y está dicho | Revisa cada `precision_score`, `recall_score`, `f1_score`. El valor por omisión de `pos_label` es 1; el de `average` es `binary` y, en multiclase, `weighted` suele coincidir con el accuracy y no aportar nada |
| A6 | Ningún hiperparámetro se eligió con la misma partición que produce la métrica reportada | Busca bucles o búsquedas de hiperparámetros y compara su `cv` con el de la evaluación final |

**De dónde salió A3.** En la Tarea 2.2 la clase positiva era el 8.88 % del conjunto: un modelo que predijera siempre "no falla" habría reportado 91 % de accuracy sin detectar una sola falla.

**De dónde salió A4 y A5.** En la Tarea 3.1 las dos versiones del modelo tenían **el mismo accuracy, 0.7343**, mientras el F1 de la clase maligna pasaba de 0.5870 a 0.6545 y los cánceres no detectados bajaban de 26 a 17. Decidir por accuracy habría dejado a nueve pacientes sin diagnóstico. En ese mismo proyecto `maligno` es la clase **0**, así que cualquier métrica escrita sin `pos_label=0` habría reportado la clase benigna con el nombre de la maligna.

**De dónde salió A6.** En la Tarea 4.1, KNN reportaba 0.9733 y aparecía como el mejor de cinco clasificadores. Su `k` se había elegido probando trece valores sobre las mismas diez particiones que luego lo evaluaban; medido con validación cruzada anidada bajó a **0.9600**, por debajo de la regresión logística. La métrica estaba bien calculada y aun así estaba inflada.

## AR. Métricas reportadas — regresión (v5)

| # | Condición | Cómo se comprueba |
|---|---|---|
| AR1 | Toda métrica está en su rango válido: MSE, RMSE y MAE ≥ 0; R² ≤ 1 | Lee los valores reportados |
| AR2 | Las métricas son coherentes entre sí: RMSE² = MSE (tolerancia 1 %) y R² = 1 − MSE / Var(y_prueba) | Recalcula con los números publicados. Si el proyecto no publica la varianza de prueba, mídela (R2) |
| AR3 | Existe un modelo de referencia trivial (`DummyRegressor` o equivalente) evaluado sobre la **misma** partición, y cada modelo reportado mejora su RMSE | Busca el Dummy y compara. Un modelo que no mejora al Dummy no aprende nada útil |
| AR4 | La métrica de decisión está elegida y argumentada para el problema: el error se interpreta **en las unidades reales** y se dice por qué una sola métrica no basta | Busca un párrafo que interprete el RMSE en unidades (dólares, días, g/km) y el R² como proporción explicada |
| AR5 | El mejor modelo tiene análisis de residuos: gráfica de residuos contra predichos, histograma, y una lectura escrita de los patrones (curvatura, dispersión creciente, grupos con sesgo) | Busca las dos gráficas y el texto que las interpreta. Gráficas sin lectura → `FALLA` (ver E4) |
| AR6 | Ningún hiperparámetro se eligió con la partición que produce la métrica reportada | Igual que A6 |

**De dónde salió AR.** De la Tarea 4.2. La v4 declaraba en su descripción que auditaba "resultados de clasificación" y la bitácora de la 4.1 lo dejó escrito: *"La Skill nunca se probó sobre un proyecto que no fuera de clasificación."* Aplicada tal cual a una regresión, A1, A2, A3 y A5 no tienen sobre qué pronunciarse, A4 pregunta por el costo del falso positivo contra el falso negativo —que no existen— y D2 mide una **caída** de la métrica, cuando en regresión el desempeño malo es un error que **sube**. El bloque AR reemplaza a A con las preguntas equivalentes.

**Cómo leer AR2 cuando falla por poco.** Una diferencia menor al 1 % es redondeo. Una mayor suele indicar que MSE y R² se calcularon sobre conjuntos distintos (uno sobre prueba y otro sobre entrenamiento), y eso sí es `FALLA`.

## B. Partición de datos

| # | Condición | Cómo se comprueba |
|---|---|---|
| B1 | La partición ocurre **antes** de cualquier transformación que aprenda de los datos (escalar, imputar, codificar por frecuencia, seleccionar características) | Compara en el rastro del paso 2 la posición de la partición contra la de cada transformación |
| B2 | Hay semilla fija en la partición y en todo estimador que la acepte | Busca `random_state` / `seed` en `train_test_split`, `KFold`, y en cada modelo |
| B3 | En clasificación, la partición es estratificada | Busca `stratify=` o `StratifiedKFold` |
| B4 | Los modelos que se comparan usan la misma partición y los mismos datos | Verifica que compartan el objeto de partición y los mismos `X`, `y` |
| B5 | A la validación cruzada se le entrega un estimador **sin entrenar** | Comprueba que no se pase un modelo al que ya se le hizo `.fit()` |
| B6 | Ninguna fila aparece a la vez en entrenamiento y prueba | Busca duplicados previos a la partición, o agrupación por sujeto cuando hay varias filas por individuo |

**De dónde salió B1.** Es el paso de la Tarea 2.2 que más tuve que verificar. En ese proyecto medir la diferencia dio **0.0000** —el conjunto era grande y homogéneo—, y eso no absuelve la práctica: **cambia lo que hay que auditar.** Un criterio del tipo "si la métrica no cambia, está bien" habría dado por buena esa versión. La condición correcta es dónde se ajusta el transformador, y eso se lee en el código, no en el número.

**De dónde salió B2.** En la Tarea 3.1 el módulo usaba un generador aleatorio global que se consumía con cada llamada. El notebook llamaba dos veces a `cargar_datos()`: exploraba un conjunto y entrenaba con otro, sin que nada avisara.

**De dónde salió B3.** También de la Tarea 3.1: la partición original no estratificaba, con clases al 37 % / 63 %.

**De dónde salió B4 y B5.** De la Tarea 3.2, donde comparé varios clasificadores y tuve que demostrar —no suponer— que todos recibían el mismo `X`, la misma `y`, la misma instancia de partición y un estimador sin entrenar.

**Cómo aplicar B6 cuando una entidad tiene varias filas (v5).** Un conjunto puede no tener duplicados exactos y aun así repetir la misma **entidad** —la misma familia en un expediente, el mismo paciente en varias consultas, la misma máquina en varias lecturas, el mismo cliente en varios meses—. Si el uso previsto es predecir sobre entidades **nuevas**, una partición aleatoria fila a fila deja la misma entidad en entrenamiento y en prueba, y la métrica mide memoria, no generalización. Mídelo (R2): qué fracción de las filas de prueba tiene su entidad también en entrenamiento, y cuántas filas de prueba tienen un vector de predictoras idéntico a alguna de entrenamiento. Si la fracción es importante y el proyecto no agrupa la partición (`GroupShuffleSplit`, `GroupKFold`), B6 es `FALLA`. Origen: Tarea 4.2.

## C. Fuga de información

| # | Condición | Cómo se comprueba |
|---|---|---|
| C1 | Ningún transformador se ajusta con datos de prueba | Todo `.fit()` / `.fit_transform()` de un preprocesador recibe solo la partición de entrenamiento, o está dentro de un `Pipeline` que la validación cruzada clona |
| C2 | Ninguna columna predictora describe algo que solo se sabría **después** de la predicción | Revisa una por una las columnas del paso 3 y pregúntate cuándo se conoce ese dato |
| C3 | Ninguna columna predictora separa **el objetivo completo** de forma perfecta o casi perfecta | Tabula cada predictora contra el objetivo. Separar perfectamente **todas** las clases a la vez es la firma de la fuga; separar perfectamente **una sola clase** es lo que hace cualquier variable buena, y no basta para marcar C3 |
| C4 | La selección de características, si existe, se ajustó solo con datos de entrenamiento | Localiza `SelectKBest`, `mutual_info`, correlaciones con el objetivo y mira qué conjunto recibieron |
| C5 | El objetivo no está entre las predictoras, ni él ni una transformación suya | Compara la lista del paso 3 contra la columna objetivo |

**De dónde salió C2 y C3.** De la Tarea 3.1. La columna `sesiones_tratamiento_programadas` tenía valores mayores que cero **solo** en las pacientes con diagnóstico maligno: se programa el tratamiento *después* de diagnosticar. El modelo alcanzaba accuracy **1.0000**. Al retirarla cayó a **0.7343**, que es el desempeño real.

**De dónde salió C4.** De la Tarea 2.2: el orden correcto y el orden con fuga daban la misma métrica en imputación y escalado, pero en **selección de características** sí se inflaba, porque elegir columnas mirando el conjunto completo decide con información de la prueba.

**Cómo aplicar C3 sin generar falsos positivos.** La separación perfecta es una señal, no un veredicto. Antes de marcar `FALLA`, cruza el resultado con C2 y responde: *¿este dato existiría antes de predecir?* Si la respuesta es sí, es una variable informativa y C3 `PASA`. En la Tarea 4.1 esta regla, escrita sin la salvedad, marcó `petal length` del conjunto Iris como fuga: separa a *setosa* sin una sola excepción (rango 1.0–1.9 contra 3.0 en adelante). No es fuga; es una medida física tomada con una regla antes de clasificar nada. Registra siempre en el informe qué columnas disparó la comprobación y cuáles descartaste, para que el lector pueda revisar tu criterio.

**Cómo aplicar C3 en regresión (v5).** No hay clases que separar. La firma equivalente es una predictora que por sí sola explica casi todo el objetivo: |correlación| ≥ 0.98, o un modelo de una sola variable con R² ≥ 0.95. Igual que en clasificación, cruza con C2 antes de marcar `FALLA`: una variable física muy informativa que existe antes de predecir no es fuga.

**Cómo aplicar C2 en la práctica.** Por cada columna predictora, responde: *¿un caso nuevo, en el momento de predecir, tendría este dato?* Si la respuesta es "solo si ya sabemos la respuesta", es fuga. Nombres que obligan a revisar: `*_posterior`, `*_final`, `*_programad*`, `*_asignad*`, `fecha_cierre`, `costo_total`, `resultado_*`.

## D. Disparidad entre subgrupos

| # | Condición | Cómo se comprueba |
|---|---|---|
| D1 | La métrica prioritaria está calculada **por subgrupo**, no solo global | Busca un desglose por las `columnas_subgrupo` |
| D2 | Ningún subgrupo con al menos 30 casos en la partición de prueba cae más de `umbral_disparidad` por debajo del valor global | Compara cada subgrupo contra el global. Si el proyecto no publica el desglose, **mídelo tú** (R2) y dilo. Solo queda en `NO SE PUEDE DETERMINAR` cuando el proyecto ni siquiera se puede ejecutar |
| D3 | El umbral está escrito y justificado en el proyecto | Busca el número y el argumento que lo sostiene |
| D4 | Los subgrupos con menos de 30 casos se reportan como no concluyentes, no como aprobados | Revisa cómo se trataron los grupos pequeños |

**Por qué 30 casos.** Por debajo de ese tamaño, un solo error mueve la métrica más de tres puntos, así que la diferencia observada no distingue entre un problema real y el azar del muestreo. El proyecto puede fijar otro tamaño mínimo; lo que no puede es omitirlo.

**Cuidado con el denominador.** Cuando la métrica prioritaria es el recall de la clase minoritaria, lo que cuenta no son los casos del subgrupo sino sus **casos positivos**: un subgrupo con 312 filas de prueba pero solo 27 fallas estima su recall sobre 27 observaciones. Reporta ambos números y, si la diferencia parece grande, confírmala con una prueba de proporciones antes de llamarla hallazgo.

**Por qué 0.10 como valor por omisión.** Es un valor de arranque, no una constante universal: marca una caída que ya es visible para un usuario del sistema sin ser tan estricta que todo subgrupo pequeño la dispare. Cada proyecto debe reemplazarlo por un umbral argumentado según el costo de su error, y el informe debe decir cuál de los dos se usó.

**D en regresión (v5).** La métrica por subgrupo es el **RMSE** (o el error que el proyecto declare como prioritario), y el fallo es que **suba**, no que caiga. D2 pasa si ningún subgrupo con al menos 30 casos de prueba tiene RMSE mayor que `umbral_disparidad` × RMSE global (1.25 por omisión). Reporta también el **residuo medio** de cada subgrupo con su signo: un RMSE parecido al global con un sesgo sistemático (el modelo siempre sobreestima o siempre subestima a ese grupo) también es un hallazgo, aunque no dispare D2.

**De dónde salió D.** De la Tarea 2.2. Al medir por grupos, las filas con valores atípicos de `temperature_c` fallaban el **25.00 %** contra el **8.74 %** de las normales. Iba a eliminarlas por verse raras; medir por subgrupo mostró que eran la señal. La lección general: un promedio global puede esconder un subgrupo en el que el modelo no funciona.

## Verificaciones complementarias

| # | Condición | Origen |
|---|---|---|
| E1 | No hay valores no finitos (`inf`, `NaN`) entrando al modelo, ni filas descartadas en silencio por ellos | Tarea 3.1: dividir entre `num_biopsias_previas` (valores 0 a 3) generaba `inf` en **140 de 569 filas**, el 24.6 % |
| E2 | Dos ejecuciones del proyecto producen los mismos números, y los números del propio proyecto son consistentes entre sí | Tarea 3.1: un generador aleatorio global hacía que dos celdas contaran la misma columna y reportaran 140 y 135 |
| E3 | Si el modelo tiene un **hiperparámetro de capacidad** (profundidad del árbol, `k` de KNN, número de estimadores, grado del polinomio, número de neuronas), hay evidencia de cómo se eligió, con el accuracy de entrenamiento y el de validación reportados juntos | Tarea 3.2: la divergencia entre ambos es lo que identifica el sobreajuste — en profundidad 5 el entrenamiento llegaba a 1.0000 y la validación bajaba a 0.9333 |
| E4 | Las afirmaciones del texto están respaldadas por una salida del propio proyecto, y las salidas que el proyecto produce están interpretadas | Transversal. En la Tarea 3.1 el notebook **graficaba** una correlación de −0.93 entre la columna con fuga y el objetivo, y ninguna celda de texto la mencionaba |

**Cómo resolver E3 sin ambigüedad.** Responde en este orden y no te saltes el primer paso:

0. En regresión, donde este procedimiento dice *accuracy* lee *R² o RMSE*. La brecha entre entrenamiento y prueba del modelo final es un síntoma, no la evidencia que pide E3: E3 pregunta cómo se **eligió** la capacidad.
1. ¿El modelo tiene algún hiperparámetro de capacidad? Si no lo tiene —una regresión logística, un Naive Bayes—, el veredicto es `NO APLICA`. `max_iter` no cuenta: controla la convergencia del optimizador, no la capacidad del modelo.
2. Si lo tiene y está escrito a mano sin ninguna medición que lo respalde, es `FALLA`. Un valor fijo sin justificación no es "no aplica": es una decisión que nadie tomó con evidencia.
3. Si lo tiene y hay una exploración, comprueba que esa exploración reporte entrenamiento **y** validación. Solo validación es `FALLA`: sin la curva de entrenamiento no se ve dónde empieza a memorizar.

# Salida

Escribe **un solo archivo**, con el nombre que indique el usuario o, si no indica ninguno, `AUDIT_REPORT_<proyecto>.md`, con esta estructura y en este orden.

```markdown
# Auditoría — <nombre del proyecto>

**Archivos auditados:** <lista con número de celdas o líneas>
**Tipo de problema:** <clasificación | regresión> (<cómo se determinó>)
**Objetivo:** `<columna>`   **Clase de interés:** `<valor>` (solo clasificación)
**Columnas de subgrupo:** <lista, o "ninguna declarada">
**Umbral de disparidad:** <valor> (<justificado por el proyecto | valor por omisión>)
**Convención de citas:** celdas numeradas desde 0 según el archivo `.ipynb`
**Fecha:** <fecha>

## Resumen

<Dos o tres frases: cuántas verificaciones fallaron, cuál es la más grave y si
las métricas del proyecto se pueden creer tal como están. Sin adjetivos que no
se sostengan en una fila de la tabla.>

## Tabla de verificación

| # | Verificación | Veredicto | Evidencia |
|---|---|---|---|
| A1 | Métricas en rango válido | PASA | celda 22: `accuracy = 0.7343` |
| ... | ... | ... | ... |

(Una fila por condición, en orden A1…A6 —o AR1…AR6 en regresión—, B1…B6, C1…C5, D1…D4, E1…E4.
La columna Evidencia lleva la referencia localizable, no una paráfrasis.)

## Hallazgos

Una subsección por cada `FALLA` y por cada `NO SE PUEDE DETERMINAR`, en orden de
gravedad. Cada una con:

**Qué encontré** — el hecho, con la cita textual del código.
**Por qué importa** — el efecto concreto sobre el número reportado.
**Qué tan seguro estoy** — confirmado por el código, o inferido y qué falta para confirmarlo.

## Acciones recomendadas

Lista numerada, ordenada por gravedad. Cada acción dice qué hacer y en qué
archivo, sin código aplicado. Formato: `<archivo/celda> — <qué cambiar> — <por qué>`.

## Lo que esta auditoría no revisó

Enumera lo que quedó fuera: archivos no leídos, condiciones que no se pudieron
comprobar y por qué. Esta sección nunca va vacía.
```

# Errores que esta Skill debe evitar en sí misma

- **Concluir `PASA` por ausencia de señal.** Si no se buscó, es `NO SE PUEDE DETERMINAR`. El paso 6 existe para atrapar esto.
- **Dejar en `NO SE PUEDE DETERMINAR` algo que sí podías medir.** Ese veredicto es para cuando la evidencia no existe y no se puede generar, no para ahorrarte una comprobación. Antes de usarlo, pregúntate si R2 te permite medirlo tú.
- **Citar la celda equivocada.** Antes de escribir una cita, vuelve al archivo y confirma el número.
- **Reportar el mismo problema como varios hallazgos.** Una causa, un hallazgo, con la lista de condiciones que arrastra.
- **Ordenar los hallazgos por el orden de las verificaciones.** Van por gravedad: primero lo que invalida las métricas, después lo que las hace frágiles, al final lo que afecta a la reproducibilidad.
- **Aplicar las condiciones de clasificación a una regresión, o al revés.** Decide el tipo de problema en el paso 1 y usa la tabla *Qué bloques aplicar*.
- **Cambiar de veredicto entre dos ejecuciones sobre el mismo proyecto.** Si ocurre, la causa suele ser una condición ambigua; reescríbela hasta que se pueda responder con un sí o un no.
