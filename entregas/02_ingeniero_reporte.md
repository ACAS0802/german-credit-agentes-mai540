# 02 · Ingeniero de modelos: árbol y Random Forest

Todos los números salen de `python experimentos.py` (versión en máquina: `02_ingeniero_metricas.json`).
scikit-learn 1.8.0, Python 3.11. Revisado tras `03_auditor_reporte.md` (ver «Respuesta a la auditoría»).

## Resultados

Referencia «siempre bueno»: exactitud 0.70, detecta 0.00 de los malos.

| Modelo | CV exactitud: media ± desv. (ddof=0 / ddof=1) | Malos detectados (recall «bad», umbral de `predict`) | Proporción predicha «bad» | Partición única 75/25 estratificada, semillas 0–9: mín – máx | Media partición única | Media − CV |
|---|---|---|---|---|---|---|
| Árbol | 0.698 ± 0.017 / 0.019 | 0.530 | 0.320 | 0.608 – 0.720 | 0.674 | −0.024 |
| Random Forest | 0.759 ± 0.006 / 0.007 | 0.377 | 0.167 | 0.712 – 0.796 | 0.753 | −0.006 |

Diferencias de los extremos de la partición única frente a la media de CV:
árbol −0.090 / +0.022; Random Forest −0.047 / +0.037.

### Variabilidad: comparación homogénea (H1)

| Modelo | Rango partición única (10) | Rango entre pliegues CV (5) | Desv. ddof=1 partición única | Desv. ddof=1 entre pliegues CV |
|---|---|---|---|---|
| Árbol | 0.112 | 0.045 | 0.031 | 0.019 |
| Random Forest | 0.084 | 0.015 | 0.028 | 0.007 |

Nota: la variabilidad que importa para la decisión es la de la **media** de CV, que con 5 pliegues
es menor que la de un solo pliegue. Ambas tablas comparan particiones individuales entre sí.

### Sensibilidad a la semilla de la CV (H3, descriptivo)

Media de CV con `StratifiedKFold(5, shuffle=True, random_state=s)`, s = 0..19 (el esquema fijo
sigue siendo `random_state=42`):

| Modelo | Mín – máx | Promedio | Semillas con media > 0.70 | Media partición única − promedio CV |
|---|---|---|---|---|
| Árbol | 0.649 – 0.712 | 0.677 | 2 / 20 | −0.003 |
| Random Forest | 0.739 – 0.774 | 0.756 | 20 / 20 | −0.003 |

### Recall según umbral (H2, descriptivo)

Sobre las probabilidades de `cross_val_predict(method="predict_proba")`. **No se eligió ningún
umbral con estos datos**; la tabla solo muestra que el recall depende del umbral.

| Modelo | Umbral | Recall «bad» | Exactitud | Proporción predicha «bad» |
|---|---|---|---|---|
| Árbol | 0.3 / 0.4 / 0.5 | 0.530 (igual) | 0.698 | 0.320 |
| Random Forest | `proba >= 0.5` | 0.403 | 0.763 | 0.179 |
| Random Forest | `proba >= 0.4` | 0.603 | 0.737 | 0.325 |
| Random Forest | `proba >= 0.3` | 0.800 | 0.686 | 0.494 |

El árbol sin límite da probabilidades 0/1, así que su recall no cambia con el umbral. La diferencia
entre 0.377 (`predict`) y 0.403 (`proba >= 0.5`) del RF se debe a 12 empates en 0.5 que `predict`
(argmax) asigna a «good».

### Lectura breve
- Con la semilla 42 el árbol sin límite queda en 0.698. Con 20 semillas de barajado su media de CV
  va de 0.649 a 0.712 (promedio 0.677) y supera 0.70 solo en 2 de 20. Queda **en torno a la
  referencia o por debajo**; no mejora la exactitud de «siempre bueno», aunque sí detecta malos.
  El Random Forest la supera con las 20 semillas.
- Con el umbral por defecto el Random Forest marca como «bad» el **16.7 %** de las solicitudes
  (el 30 % real) y detecta 0.377 de los malos, frente a 0.530 del árbol. **Esta comparación de
  recall solo vale para ese umbral**: no es una propiedad fija del modelo. Con umbral 0.4 el RF
  detectaría 0.603 con exactitud 0.737. Para fijar un umbral habría que elegirlo con datos
  distintos de los de evaluación (p. ej. CV anidada); no lo hice.
- Una sola partición 75/25 varía más que un pliegue de CV: rango 0.112 frente a 0.045 en el árbol
  (desv. ddof=1: 0.031 frente a 0.019), y 0.084 frente a 0.015 en el RF. Con una sola partición el
  árbol puede aparecer en 0.720 (por encima de la referencia) o en 0.608.
- La «Media − CV» del árbol (−0.024) no indica un sesgo de la partición única: frente al promedio de
  CV sobre 20 semillas la diferencia es −0.003. La semilla 42 cae del lado alto.

## Decisiones tomadas (y por qué)

1. **Pipeline** (`modelos.py`): `Pipeline([("prep", ColumnTransformer), ("modelo", clasificador)])`.
   - `prep`: `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` sobre las 13
     categóricas de `cols_cat`, siguiendo la advertencia 3 del analista (categorías raras).
   - Las 7 numéricas pasan con `remainder="passthrough"`, sin escalar: los modelos son árboles.
   - Nombres `prep` y `modelo` porque `tests/` y `busqueda.py` (`modelo__…`) dependen de ellos.
2. **Modelos** (`tres_modelos`): hiperparámetros por defecto y `random_state=42` en los tres
   (`DecisionTreeClassifier`, `RandomForestClassifier` con 100 árboles, `HistGradientBoostingClassifier`).
   El árbol se deja sin límite de profundidad a propósito: la prueba `test_supera_a_siempre_bueno`
   marca el árbol como `xfail` y `busqueda.py` es quien explora `max_depth`. Boosting no se evaluó
   aquí (fuera del alcance pedido).
3. **Validación cruzada**: se usa sin cambios el `cv` de `datos.py`
   (`StratifiedKFold(5, shuffle=True, random_state=42)`). `validar` devuelve la desviación con
   **ddof=0** (`numpy.std`); no la cambié para no alterar la salida de `main.py`. El JSON y la tabla
   dan también **ddof=1** (`cv_pliegues.desviacion_ddof1`).
4. **Malos detectados**: `cross_val_predict` con el mismo `cv` y `recall_score(pos_label=1)`, con el
   umbral implícito de `predict` (argmax).
5. **Partición única**: `train_test_split(test_size=0.25, stratify=y, random_state=s)`, s = 0..9
   (750/250). Decidí estratificar para que sea coherente con la CV, que también estratifica. El
   efecto de no estratificar **no es uniforme**: con las semillas 0..9 el rango sube en el árbol
   (0.128 frente a 0.112) y baja en el RF (0.068 frente a 0.084). Datos en
   `particion_unica_no_estratificada` del JSON. La semilla del modelo queda en 42.
6. En todas las funciones de `validacion.py` se usa `clone(pipeline)` para no reutilizar un
   modelo ya ajustado.

## Comando para reproducir

```bash
cd german-credit-agentes-mai540
python experimentos.py      # ~35 s; imprime el resumen y escribe entregas/02_ingeniero_metricas.json
```

## Para que revisen

**Auditor**
- Las secciones nuevas del JSON (`comparacion_homogenea`, `sensibilidad_umbral_descriptiva`,
  `sensibilidad_semilla_cv`, `particion_unica_no_estratificada`) son descriptivas; las métricas
  principales no cambiaron.
- Si se quiere que `validar()` devuelva ddof=1, es un cambio de una línea en `validacion.py` que
  cambiaría la salida de `main.py`; lo dejo a decisión del coordinador.

**Analista**
- Ninguna objeción al esquema de `datos.py`. La sensibilidad a la semilla se calculó aparte en
  `experimentos.py`, sin modificar el `cv` fijo.

## Respuesta a la auditoría

| Hallazgo | Respuesta | Cambio |
|---|---|---|
| **H1** · rango comparado con desviación | De acuerdo. | Quité la comparación rango vs. desviación. Nueva tabla «Variabilidad: comparación homogénea» (rango vs. rango, desv. ddof=1 vs. desv. ddof=1) y bloque `comparacion_homogenea` en el JSON, por modelo. Los valores coinciden con los del auditor. |
| **H2** · «casi todo como bueno» y recall dependiente del umbral | De acuerdo con (a) y (b). | (a) Doy la cifra: el RF predice «bad» en el 16.7 % (`proporcion_predicha_bad`). (b) Indico que el recall es con el umbral de `predict` y que la comparación con el árbol solo vale para ese umbral. Añadí la tabla y `sensibilidad_umbral_descriptiva` (0.3, 0.4 y 0.5), explicando los empates en 0.5. No elegí umbral con los datos de evaluación. |
| **H3** · «no supera 0.70» depende de la partición fija | De acuerdo: mi nota iba en la dirección equivocada. | La reemplacé: con 20 semillas el árbol supera 0.70 en 2 de 20 (promedio 0.677). Añadí `sensibilidad_semilla_cv` y la lectura de que −0.024 no es un sesgo de la partición única (−0.003 frente al promedio). El esquema fijo de `datos.py` no cambió. |
| **H5** · matizar estratificación e indicar ddof | De acuerdo. | La decisión 5 ahora dice que sin estratificar el rango sube en el árbol y baja en el RF, y el JSON incluye `particion_unica_no_estratificada`. La cabecera de la tabla indica ddof=0 / ddof=1. En el JSON añadí `cv_exactitud_desviacion_ddof: 0` y `desviacion_ddof0`/`ddof1` en cada resumen. |
| H4 | Dirigido al analista. | No toco `01_*`. Mi lectura del árbol usa la formulación que propone el auditor. |
