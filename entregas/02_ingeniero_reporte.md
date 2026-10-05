# 02 · Ingeniero de modelos: árbol y Random Forest

Todos los números salen de `python experimentos.py` (versión en máquina: `02_ingeniero_metricas.json`).
scikit-learn 1.8.0, Python 3.11.

## Resultados

Referencia «siempre bueno»: exactitud 0.70, detecta 0.00 de los malos.

| Modelo | CV exactitud (media ± desv.) | Malos detectados (recall «bad») | Partición única 75/25, semillas 0–9: mín – máx | Rango | Media partición única | Media − CV |
|---|---|---|---|---|---|---|
| Árbol | 0.698 ± 0.017 | 0.530 | 0.608 – 0.720 | 0.112 | 0.674 | −0.024 |
| Random Forest | 0.759 ± 0.006 | 0.377 | 0.712 – 0.796 | 0.084 | 0.753 | −0.006 |

Diferencias de los extremos de la partición única frente a la media de CV:
árbol −0.090 / +0.022; Random Forest −0.047 / +0.037.

Lectura breve:
- El árbol sin límite no supera la referencia de 0.70 en CV (0.698); el Random Forest sí (0.759).
- El Random Forest detecta menos malos (0.38) que el árbol (0.53): gana exactitud a costa de
  marcar casi todo como «bueno». Con el desbalance 70/30 esto importa.
- Una sola partición puede dar desde 0.608 hasta 0.720 para el mismo árbol: el rango (0.112) es
  mucho mayor que la desviación entre pliegues de CV. Una sola partición puede hacer que el árbol
  parezca mejor que la referencia (0.720) o claramente peor (0.608).

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
   (`StratifiedKFold(5, shuffle=True, random_state=42)`). `validar` devuelve `std` poblacional
   (`numpy.std`, ddof=0) de los 5 pliegues.
4. **Malos detectados**: `cross_val_predict` con el mismo `cv` y `recall_score(pos_label=1)`.
5. **Partición única**: `train_test_split(test_size=0.25, stratify=y, random_state=s)`, s = 0..9
   (750/250). **Decidí estratificar** para que la variación venga del muestreo y no de cambios en la
   proporción de malos; sin estratificar el rango probablemente sería algo mayor. La semilla de la
   partición cambia; la del modelo queda en 42.
6. En todas las funciones de `validacion.py` se usa `clone(pipeline)` para no reutilizar un
   modelo ya ajustado.

## Comando para reproducir

```bash
cd german-credit-agentes-mai540
python experimentos.py      # imprime la tabla y escribe entregas/02_ingeniero_metricas.json
```

## Para que revisen

**Auditor**
- Que el codificador se ajuste solo con el entrenamiento de cada pliegue (está dentro del
  Pipeline, pero la verificación de fuga es suya).
- Si `std` con ddof=0 es lo que se quiere reportar, o se prefiere ddof=1.
- Si la estratificación de la partición única es aceptable para la comparación con CV
  (decisión 5).
- El recall de «bad» del Random Forest (0.38) con umbral 0.5: la exactitud alta no implica que
  detecte bien los malos.

**Analista**
- Ninguna objeción al esquema de `datos.py`. Solo una nota: el árbol queda en 0.698, a 0.002 de la
  referencia, de modo que la conclusión «no supera 0.70» depende de la partición de CV fija.
