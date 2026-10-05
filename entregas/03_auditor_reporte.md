# 03 · Auditor: verificación de los resultados del ingeniero

No modifiqué código ni entregables de otros roles. Las comprobaciones se hicieron en una copia
temporal (`/tmp/claude-0/auditor/repo/`, con `datos/`, `datos.py`, `modelos.py`, `validacion.py`,
`experimentos.py`) y con un script propio (`/tmp/claude-0/auditor/checks.py`, salida en
`/tmp/claude-0/auditor/checks_out.txt`). Entorno: Python 3.11.15, scikit-learn 1.8.0 (coincide
con lo declarado en `02_ingeniero_reporte.md:4`).

## Tabla de veredictos

| # | Punto | Veredicto | Evidencia |
|---|---|---|---|
| 1 | Fuga | **PASA** | `modelos.py:11-16`: `OrdinalEncoder` dentro de `ColumnTransformer` dentro del `Pipeline`; `validacion.py:10,16,26` pasan el pipeline completo a `cross_val_score` / `cross_val_predict` / `fit`. `datos.py:10` quita `clase` de `X` (comprobado: `'clase' in X: False`). Con `cross_validate(return_estimator=True)`, en los 5 pliegues `encoder.categories_` es exactamente el conjunto de categorías del entrenamiento del pliegue (`categorias==train:True` ×5). Control: ajustando el codificador con todo `X` (fuga deliberada) se obtiene lo mismo (0.698 / 0.759), porque toda categoría aparece en cada entrenamiento de CV y en las 10 particiones 75/25 (`faltan_en_train={}`); es decir, no hay fuga y, aunque la hubiera, con este codificador no inflaría la exactitud. |
| 2 | Clase positiva | **PASA** | `datos.py:9`: `y = (clase == "bad")`; `validacion.py:17`: `recall_score(y, pred, pos_label=1)`. Recalculado con etiquetas de texto: `recall_score(..., pos_label="bad")` = 0.53 (árbol) y 0.3767 (RF); el recall de «good» sería 0.77 / 0.9229, distinto del reportado. Matrices de confusión (CV): árbol `[[539,161],[141,159]]` → 159/300 = 0.530; RF `[[646,54],[187,113]]` → 113/300 = 0.3767. |
| 3 | Esquema | **PASA** | `datos.py:12`: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, usado sin cambios (`experimentos.py:20,24-25`). 5 pliegues 800/200 con proporción de «bad» 0.30 en cada uno. Partición única: `validacion.py:23-25` + `experimentos.py:15` → `test_size=0.25`, `stratify=y`, semillas 0..9 (10 semillas). |
| 4 | Reproducibilidad | **PASA** | `cd /tmp/claude-0/auditor/repo && python experimentos.py` → `Árbol CV 0.698 ± 0.017 \| recall bad 0.530 \| partición única 0.608–0.720 (media 0.674)` y `Random Forest CV 0.759 ± 0.006 \| recall bad 0.377 \| partición única 0.712–0.796 (media 0.753)`. El JSON regenerado es **idéntico byte a byte** al entregado (`sha256 2387421…5724` en ambos; `diff` vacío). |
| 5 | Coherencia | **PASA** | Cada celda de la tabla (`02_ingeniero_reporte.md:12-13`) y cada número de la prosa (líneas 8, 16, 19-24) coincide con `02_ingeniero_metricas.json` redondeado (p. ej. 0.0169→0.017, 0.3767→0.377/0.38, −0.0236→−0.024, extremos −0.090/+0.022 y −0.047/+0.037). La referencia 0.70 coincide con el analista (`01_analista_datos.json`: 1000 filas, 300 «bad», 0.3) y con el JSON del ingeniero (`referencia_siempre_bueno: 0.7`). Todos los números del analista que comprobé se sostienen (filas, 13/7 columnas, numéricas enteras, rangos y medias, categorías y mínimos, «domestic appliance» y «other» n=12, 0 faltantes, 0 duplicadas, subgrupos y sus n). |
| 6 | Interpretación | **FALLA** (parcial) | Las conclusiones principales están respaldadas (RF 0.759 supera 0.70 en los 5 pliegues y en las 10 particiones; el árbol 0.698 no la supera). Pero hay tres afirmaciones que los números no sostienen tal como están escritas: comparación rango vs. desviación (H1), «marcar casi todo como bueno» y la comparación de recall presentada como intrínseca cuando depende del umbral (H2), y la nota de que «no supera 0.70» depende de la partición fija (H3). Además, una afirmación del analista contradice los resultados (H4). Ver hallazgos. |

## Decisiones declaradas por el ingeniero

| Decisión | ¿Aceptable? | ¿Cambia la comparación pedida? | Evidencia |
|---|---|---|---|
| Partición 75/25 **estratificada** | Sí: es coherente con la CV, que también estratifica, así que se comparan esquemas que solo difieren en «una partición vs. cinco pliegues». | No en lo esencial. Sin estratificar (semillas 0..9): árbol 0.616–0.744 (rango 0.128), RF 0.724–0.792 (rango 0.068). Con 100 semillas: desv. (ddof=1) árbol 0.0288 estrat. vs 0.0314 no estrat.; RF 0.0230 vs 0.0217. La frase «sin estratificar el rango probablemente sería algo mayor» (`02_ingeniero_reporte.md:44`) se cumple para el árbol pero **no** para el RF con las semillas 0..9; debe matizarse. | `checks_out.txt`, sección «particion unica» |
| `std` con **ddof=0** | Aceptable si se declara (lo declara en `02_ingeniero_reporte.md:39-40`). Para 5 pliegues tratados como muestra, ddof=1 es lo habitual. | No: árbol 0.0169→0.0189, RF 0.0058→0.0065; el orden y las conclusiones no cambian. Recomiendo reportar ddof=1 o indicarlo en la cabecera de la tabla. | `validacion.py:11`; `checks_out.txt`, sección «ddof» |
| Árbol **sin límite de profundidad** | Sí: son los valores por defecto (`modelos.py:22`) y es la lección prevista (`tests/test_modelo.py:13`, `xfail` «el árbol sin límite no supera el 0.70»; `busqueda.py:6` explora `max_depth`). | No; es precisamente el árbol que se pide comparar. Nota: la prueba `test_supera_a_siempre_bueno` aún es un `TODO` que lanza `NotImplementedError` (`tests/test_modelo.py:16-17`), así que hoy no verifica nada; no se pudo ejecutar `pytest` (no instalado en el entorno). | — |
| **`clone`** | Sí. | No. Tras `validar(pipe, ...)` el pipeline original sigue sin ajustar (`NotFittedError`), y cada repetición de la partición única usa una copia limpia (`validacion.py:10,16,26`). | `checks_out.txt`, sección «clone» |
| **Umbral 0.5** (`predict`) | Aceptable como valor por defecto, pero el recall del RF depende mucho de él. | **Sí cambia la lectura del recall.** El árbol sin límite da probabilidades 0/1, así que su recall es 0.53 con cualquier umbral. El RF, con probabilidades de CV: umbral 0.4 → recall 0.603, exactitud 0.737; umbral 0.3 → recall 0.800, exactitud 0.686. Con 0.4 el RF supera al árbol **en ambas** métricas. Detalle técnico: `predict` usa `argmax`, de modo que los 12 casos con probabilidad exactamente 0.5 (8 de ellos «bad») van a «good»; con `proba >= 0.5` el recall sería 0.403. | `checks_out.txt`, sección «CLASE POSITIVA / recall / umbral» |

## Hallazgos

**H1 · Para el ingeniero de modelos (interpretación).** `02_ingeniero_reporte.md:22-23` compara el
**rango** de 10 particiones (0.112) con la **desviación estándar** de 5 pliegues (0.017). Son
estadísticos distintos y la comparación exagera la diferencia. Comparación homogénea: rango de la
partición única 0.112 vs. rango entre pliegues de CV 0.045 (árbol; RF 0.084 vs 0.015), o desv.
(ddof=1) de la partición única 0.031 vs. 0.019 entre pliegues (RF 0.028 vs 0.0065). La conclusión
cualitativa (una sola partición varía más) se mantiene, pero debe reescribirse con estadísticos
comparables.

**H2 · Para el ingeniero de modelos (interpretación).** `02_ingeniero_reporte.md:20-21`: «gana
exactitud a costa de marcar casi todo como "bueno"».
(a) El RF marca como «bad» el 16.7 % de las solicitudes (frente al 30 % real), es decir, el 83.3 % como
«bueno»; «casi todo» exagera. Hay que dar la cifra.
(b) La comparación «el RF detecta menos malos que el árbol» solo vale con umbral 0.5. Con umbral 0.4
el RF detecta 0.603 de los malos con exactitud 0.737, y supera al árbol en las dos métricas. Debe
indicarse que el recall reportado es con el umbral 0.5 y que no es una propiedad fija del modelo. Si se
elige un umbral, debe elegirse sin usar los mismos datos de evaluación.

**H3 · Para el ingeniero de modelos (interpretación).** `02_ingeniero_reporte.md:68-69` dice que
«no supera 0.70» depende de la partición de CV fija. Con 20 semillas de `StratifiedKFold` (0..19), la
media de CV del árbol va de 0.649 a 0.712 (promedio 0.677) y supera 0.70 solo en 2 de 20. La
conclusión es más robusta de lo que sugiere la nota, y la semilla 42 (0.698) cae del lado favorable.
Lo correcto es decir que el árbol queda en torno a la referencia o por debajo, no que se le gana por
0.002. De paso, esto explica la mayor parte de «Media − CV = −0.024» del árbol (media de las
particiones 0.674 frente a 0.677 de CV promediada sobre semillas), que no debe leerse como un sesgo
de la partición única.

**H4 · Para el analista de datos (afirmación no respaldada).** `01_analista_resumen.md:9`: «Un
modelo con exactitud ≤ 0.70 no aporta nada». Es falso con los propios resultados: el árbol, con
exactitud 0.698, detecta el 53 % de los malos, frente al 0 % de «siempre bueno». Contradice además
la advertencia 1 del mismo documento. Reformular: «con exactitud ≤ 0.70 no mejora la exactitud de la
referencia; hay que mirar también los malos detectados».

**H5 · Para el ingeniero de modelos (menor).** `02_ingeniero_reporte.md:43-44`: matizar «sin
estratificar el rango probablemente sería algo mayor». Con las semillas 0..9 eso ocurre en el árbol
(0.128 > 0.112) pero no en el RF (0.068 < 0.084). Indicar también ddof en la cabecera de la tabla.

Ninguno de los hallazgos cambia los números del JSON ni el esquema de validación. Solo se deben
corregir las frases señaladas de `02_ingeniero_reporte.md` (H1, H2, H3, H5) y de
`01_analista_resumen.md` (H4).
