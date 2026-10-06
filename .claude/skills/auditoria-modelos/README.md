# Skill: `auditoria-modelos`

Auditoría de proyectos de clasificación y, desde la v5, de regresión, antes de creerles sus métricas.

## Qué audita

Veinticinco condiciones agrupadas en cinco bloques. Cada una se responde con `PASA`, `FALLA`, `NO SE PUEDE DETERMINAR` o `NO APLICA`, y cada respuesta cita la celda o la línea donde está la evidencia.

| Bloque | Qué revisa | Condiciones |
|---|---|---|
| **AR. Métricas de regresión (v5)** | Rango válido de MSE/RMSE/R², coherencia RMSE² = MSE y R² = 1 − MSE/Var, modelo de referencia trivial sobre la misma partición, error interpretado en unidades reales, análisis de residuos, e hiperparámetros no elegidos con la partición que evalúa | AR1–AR6 |
| **A. Métricas** | Rango válido, consistencia con la matriz de confusión, que no se reporte solo accuracy con clases desbalanceadas, que la métrica prioritaria corresponda al costo del error, que `pos_label` (o el promedio, en multiclase) apunte a la clase correcta, y que ningún hiperparámetro se haya elegido con la partición que produce la métrica | A1–A6 |
| **B. Partición** | Que la división ocurra antes de todo preprocesamiento que aprenda de los datos, semilla fija, estratificación, misma partición entre modelos comparados, estimador sin entrenar en la validación cruzada, y sin filas compartidas entre train y test | B1–B6 |
| **C. Fuga** | Que ningún transformador se ajuste con datos de prueba, que ninguna predictora describa algo posterior a la predicción, separación perfecta, selección de características ajustada solo con train, y que el objetivo no esté entre las predictoras | C1–C5 |
| **D. Subgrupos** | Métrica desglosada por subgrupo, caída máxima tolerada, umbral justificado, y grupos pequeños marcados como no concluyentes | D1–D4 |
| **E. Complementarias** | Valores no finitos, reproducibilidad y consistencia interna de los números, train contra validación al elegir capacidad, y afirmaciones respaldadas por salidas | E1–E4 |

## Cómo se invoca

```
/auditoria-modelos
```

O describiendo la tarea, que es como la Skill se carga sola:

> Audita `App_Diagnostico_Biopsias_Mama.ipynb`. El objetivo es `diagnostico`, la clase de interés es `0` (maligno) y los subgrupos son `num_biopsias_previas`.

**Da siempre estos tres datos**, porque la Skill los pide y no los adivina:

1. Qué archivos auditar
2. Cuál es la columna objetivo y **cuál es la clase de interés** (no asume que sea `1`)
3. Qué columnas definen subgrupos — si no declaras ninguna, la verificación D se resuelve `NO SE PUEDE DETERMINAR`, nunca `PASA`

## Cómo interpretar el informe

El informe es `AUDIT_REPORT_<proyecto>.md` y se lee en este orden:

1. **El resumen** dice si las métricas del proyecto se pueden creer tal como están.
2. **La tabla de verificación** da el detalle condición por condición. La columna *Evidencia* es lo que hace auditable la auditoría: si una cita no se puede seguir hasta el código, ese veredicto no vale.
3. **Los hallazgos** van ordenados por gravedad, no por número de verificación. El primero es el que más afecta a los números reportados.
4. **Lo que esta auditoría no revisó** nunca va vacío. Léelo antes de dar el proyecto por aprobado.

Los cuatro veredictos no son una escala. `NO SE PUEDE DETERMINAR` **no** es un aprobado con reservas: significa que la evidencia no existe en el código, y es un resultado legítimo que a menudo señala exactamente lo que falta documentar. `NO APLICA` es distinto: la condición no viene al caso en este proyecto (una condición binaria en un problema multiclase, por ejemplo) y no es un reproche.

## Limitaciones conocidas

- **Es una auditoría estática.** Lee código y salidas; no reentrena modelos ni recalcula métricas por su cuenta. Detecta que un procedimiento es incorrecto, no cuánto se infló el número por serlo.
- **Depende de que la clase de interés se declare bien.** Si se le dice que la clase de interés es `1` cuando es `0`, la auditoría dará por buenas métricas equivocadas. Es el error más caro que puede cometer, y no tiene cómo detectarlo sola.
- **La verificación D necesita que los subgrupos existan como columnas.** Un sesgo por una variable que el proyecto no registró es invisible para esta Skill.
- **C2 exige criterio humano.** Decidir si una columna se conoce antes o después de la predicción requiere entender el dominio. La Skill señala nombres sospechosos y separaciones perfectas, pero puede omitir una fuga sutil cuyo nombre de columna sea inocente.
- **C3 tiende al falso positivo.** Una variable muy informativa puede separar perfectamente **una** clase sin que haya fuga alguna. La v2 exige cruzar C3 con C2 antes de marcar el fallo, pero el juicio final sigue siendo humano.
- **No audita el conjunto de datos.** No revisa cómo se recolectó, a quién representa ni si las etiquetas son correctas.
- **Los umbrales por omisión son puntos de partida.** 0.10 de disparidad y 30 casos mínimos son valores razonables para empezar, no constantes del dominio.

## Historial

| Versión | Cambio | Qué lo provocó |
|---|---|---|
| v1 | Cuatro bloques (A–D), diecinueve condiciones, derivadas de los errores de los Módulos 1 a 3 | — |
| v2 | **+A5** (`pos_label` y promedio en multiclase), **+A6** (hiperparámetro elegido con la partición que evalúa), **+bloque E** (E1–E4), **+veredicto `NO APLICA`**, y **C3 reescrita** para exigir separación perfecta del objetivo completo, no de una sola clase | Ejecutar la Skill sobre los tres proyectos. La auditoría de Iris dejó a la vista un falso positivo de C3 (`petal length` separa a *setosa*) y un hueco en A5 (multiclase). La del proyecto con defectos inyectados mostró que sin el bloque E se escapaban la no reproducibilidad y las conclusiones sin respaldo |
| v4 | **E3 reescrita** como una pregunta de tres pasos: primero si el modelo tiene hiperparámetro de capacidad (si no, `NO APLICA`), después si está justificado (si no, `FALLA`), y por último si la exploración reporta entrenamiento y validación | **Veredicto inestable detectado.** La misma condición dio `FALLA` sobre el proyecto con defectos inyectados (`max_depth=3` fijo) y `NO APLICA` sobre el de mantenimiento predictivo (ningún parámetro de capacidad), ante situaciones que el texto de v3 no distinguía. La redacción no permitía responder con un sí o un no |
| v5 | **Soporte para regresión.** Nueva entrada `tipo_de_problema`; tabla que indica qué bloques aplicar; **bloque AR** (AR1–AR6) en lugar de A; B3 `NO APLICA` en regresión; nota de B6 sobre **entidades repetidas** (misma entidad en varias filas); C3 leída como correlación o R² de una sola variable; D2 con **RMSE que sube** y umbral por omisión 1.25 × global; E3 con R²/RMSE; nombre del informe elegible por el usuario | Tarea 4.2. La v4 declaraba alcance de clasificación y su propia bitácora advertía que nunca se había probado en regresión. Aplicada tal cual a la herramienta de duración de casos migratorios, A1–A5 no tenían sobre qué pronunciarse y D2 medía el fallo en la dirección equivocada |
| v3 | **R2 aclarada**: la Skill puede tomar mediciones propias para sustentar un veredicto, rotulándolas como tales, sin escribir dentro del proyecto. **D2 reescrita** en consecuencia, y nota sobre el denominador cuando la métrica es el recall de una clase minoritaria | La auditoría del proyecto de mantenimiento predictivo. La v2 dejaba D2 en `NO SE PUEDE DETERMINAR` por una lectura estricta de R2, cuando la disparidad sí se podía medir — y al medirla apareció el hallazgo más grave de las cuatro auditorías: el recall cae de 0.6306 a 0.3704 en el subgrupo `environment = humedo` (Fisher exacta, p = 0.0024) |
