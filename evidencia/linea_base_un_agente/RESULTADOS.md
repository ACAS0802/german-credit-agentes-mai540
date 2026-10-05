# Resultados · Árbol de decisión frente a Random Forest (German Credit)

Datos: 1,000 solicitudes, 20 variables (13 categóricas), 30 % de créditos malos.
Clase positiva: `bad` → 1. Validación cruzada: `StratifiedKFold(5, shuffle=True, random_state=42)`.
Partición única: 75/25 estratificada con semillas 0–9. Se reproduce con `python resultados.py`.

## Tabla de resultados

| Modelo | Exactitud CV (media ± desv.) | Malos detectados (recall de `bad`) | Partición única: mín | máx | rango | mín − CV | máx − CV |
|---|---|---|---|---|---|---|---|
| Árbol | 0.698 ± 0.017 | 0.530 | 0.608 | 0.720 | 0.112 | −0.090 | +0.022 |
| Random Forest | 0.759 ± 0.006 | 0.377 | 0.712 | 0.796 | 0.084 | −0.047 | +0.037 |

Media de las 10 particiones únicas: árbol 0.674, Random Forest 0.753.
Referencia: predecir siempre "bueno" da 0.700 de exactitud y detecta 0 malos.

## Interpretación

- **El árbol sin límite no supera la referencia trivial** (0.698 frente a 0.700). Se memoriza el
  entrenamiento y generaliza mal. El Random Forest sí la supera, por unos 6 puntos, y su resultado varía
  menos entre pliegues (desv. 0.006 frente a 0.017).
- **Una sola partición engaña.** Según la semilla, el árbol obtiene entre 0.608 y 0.720 (11 puntos de rango).
  Con una partición afortunada se podría informar que "supera el 70 %", y con una desafortunada,
  que es 9 puntos peor de lo que es. Con el Random Forest el rango es de 8 puntos. La validación cruzada
  promedia 5 evaluaciones con todos los casos y da una cifra más estable. Aun así, una desviación entre pliegues de
  0.006 subestima la incertidumbre real: las particiones únicas muestran que, con 250 casos de prueba,
  ±4 puntos es ruido normal.
- **Exactitud no es lo mismo que detectar malos.** El Random Forest tiene mejor exactitud, pero detecta solo el 38 %
  de los créditos malos, frente al 53 % del árbol. Gana exactitud sobre todo porque acierta más créditos buenos,
  que son la mayoría. Si lo costoso es aprobar un crédito malo, ningún modelo con el umbral por defecto (0.5) es suficiente.

## Conclusiones

1. Para exactitud y estabilidad, el Random Forest es claramente mejor que el árbol sin podar.
2. No hay que informar un resultado a partir de una sola partición 75/25. Con la validación cruzada estratificada
   se obtiene una media y una dispersión.
3. Para usar el modelo en un caso real, hay que optimizar una métrica centrada en los malos (recall,
   coste asimétrico) o ajustar el umbral o los pesos de clase. La exactitud sola no basta.

## Autoverificación

- **Fuga de información:** el `OrdinalEncoder` está dentro del `Pipeline` (pasos `prep` → `modelo`), así que
  se ajusta solo con los datos de entrenamiento de cada pliegue o partición. `particion_unica` usa `clone` y entrena
  solo con el 75 %. El recall se calcula con `cross_val_predict`, así que cada caso lo predice un modelo que no lo vio. Las categorías
  que no aparecen en el entrenamiento se codifican como −1 (`handle_unknown`) y no producen errores. Correcto.
- **Etiqueta positiva:** comprobé con una tabla cruzada que `bad` → 1 (300 casos) y `good` → 0 (700 casos).
  `recall_score(..., pos_label=1)` mide entonces la proporción de malos detectados. Correcto.
- **Reproducibilidad:** todos los modelos usan `random_state=42`, la CV usa la semilla 42 y las particiones usan las semillas 0–9.
  Ejecuté `resultados.py` dos veces y la salida fue idéntica byte a byte (mismo md5). Correcto.
- **Limitaciones:** no pude ejecutar `pytest` porque no está instalado en este entorno. Verifiqué a mano que el
  pipeline tiene el paso `prep` y que el Boosting funciona (CV 0.752 ± 0.021). Los TODO 7–12 no formaban parte de esta tarea,
  así que `main.py` todavía falla en la sección 3.
