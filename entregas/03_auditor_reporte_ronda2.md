# 03 · Auditor, ronda 2: verificación de las respuestas a H1–H5

No modifiqué ningún archivo salvo este. `03_auditor_reporte.md` (ronda 1) se mantiene sin cambios.

## Reproducibilidad (desde cero, copia nueva)

- Copia nueva en `/tmp/claude-0/auditor/ronda2/` con `datos/`, `datos.py`, `modelos.py`, `validacion.py`
  y el `experimentos.py` actualizado. Comando `python experimentos.py` (32.8 s). Salida:
  `Árbol CV 0.698 ± 0.019 (ddof=1) | recall bad 0.530 | predice bad 0.320 | partición única 0.608–0.720 | CV semillas 0–19 0.649–0.712`
  y `Random Forest CV 0.759 ± 0.007 (ddof=1) | recall bad 0.377 | predice bad 0.167 | partición única 0.712–0.796 | CV semillas 0–19 0.739–0.774`.
- `cmp` contra `entregas/02_ingeniero_metricas.json`: **idéntico byte a byte**. El sha256 de ambos es
  `262c4e19…fbd82`. **PASA.**
- `datos.py`, `modelos.py` y `validacion.py` son idénticos byte a byte a los auditados en la ronda 1
  (`cmp`). Solo cambió `experimentos.py`, y sus añadidos no alteran el pipeline ni el `cv` fijo. Las
  métricas principales del JSON (0.698 / 0.0169 / 0.53 / particiones del árbol; 0.759 / 0.0058 /
  0.3767 / particiones del RF) son las mismas de la ronda 1. Los veredictos 1–5 de la ronda 1 se
  mantienen.

## Estado de los hallazgos

| Hallazgo | Rol | Estado | Evidencia |
|---|---|---|---|
| **H1** · rango comparado con desviación | Ingeniero | **CERRADO** | `02_ingeniero_reporte.md:18-26` y `:64-66` ya comparan rango con rango (0.112 vs 0.045; 0.084 vs 0.015) y desv. ddof=1 con desv. ddof=1 (0.031 vs 0.019; 0.028 vs 0.007). Coincide con el JSON (`comparacion_homogenea`) y con mis cálculos de la ronda 1. La nota sobre la variabilidad de la *media* de CV es correcta. |
| **H2** · «casi todo como bueno» / recall dependiente del umbral | Ingeniero | **CERRADO** | (a) `:59-60` da la cifra: 16.7 % predicho «bad» (`proporcion_predicha_bad: 0.167`; mi matriz de confusión de la ronda 1 da (54+113)/1000 = 0.167). (b) `:60-63` dice que la comparación solo vale para el umbral de `predict`, no elige umbral con los datos de evaluación y remite a CV anidada. La tabla `:43-48` coincide con el JSON y con mis valores (0.403/0.763/0.179; 0.603/0.737/0.325; 0.800/0.686/0.494), y explica bien los 12 empates en 0.5 (8 de ellos «bad»), que yo había verificado. |
| **H3** · «no supera 0.70» y la semilla fija | Ingeniero | **CERRADO** | `:28-36` y `:55-58`, `:67-68`: árbol 0.649–0.712, promedio 0.677, 2/20 > 0.70. Es idéntico a mi cálculo independiente de la ronda 1 (`/tmp/claude-0/auditor/checks_out.txt`). La lectura («en torno a la referencia o por debajo»; −0.024 no es sesgo de la partición única, −0.003 frente al promedio) está respaldada. La respuesta antigua (`:68-69` de la ronda 1) fue retirada. El RF (0.739–0.774, 20/20) se reproduce. |
| **H4** · «exactitud ≤ 0.70 no aporta nada» | Analista | **CERRADO** | `01_analista_resumen.md:9` dice ahora: «no mejora la exactitud de la referencia; hay que mirar también la proporción de malos detectados». La nota de corrección está en `:37`. `grep "aporta" entregas/0[12]*` solo encuentra la nota de corrección. `01_analista_datos.json` sin cambios. |
| **H5** · matizar estratificación e indicar ddof | Ingeniero | **CERRADO** | La decisión 5 (`:88-92`) dice que el rango sube en el árbol (0.128 vs 0.112) y baja en el RF (0.068 vs 0.084). Coincide con `particion_unica_no_estratificada` y con mis valores de la ronda 1. La cabecera de la tabla (`:10`) indica ddof=0 / ddof=1, y el JSON añade `cv_exactitud_desviacion_ddof: 0`. Dejar `validar()` en ddof=0 para no cambiar `main.py` es aceptable porque está declarado. |

## Observaciones nuevas sobre los cambios (menores, no bloquean)

**N1 · Para el ingeniero de modelos: cambio de esquema del JSON.** En `particion_unica`, la clave
`exactitudes` de la ronda 1 se llama ahora `valores` (`experimentos.py:36,101`). Ningún archivo del
repositorio la lee (`grep -rn exactitudes` solo aparece en `validacion.py`, como variable local), así
que hoy no rompe nada. Aun así, cualquiera que leyera la versión anterior del JSON fallaría. Conviene
mencionarlo en «Respuesta a la auditoría» o conservar el nombre original.

**N2 · Para el ingeniero de modelos: desviación mostrada en pantalla y en el JSON.** La línea impresa
(`experimentos.py:121`) muestra `±` con ddof=1 (0.019 / 0.007). El campo principal del JSON
`cv_exactitud_desviacion` sigue con ddof=0 (0.0169 / 0.0058), y `main.py` (vía `validar()`) también
usa ddof=0. Las tres salidas están etiquetadas, así que no es un error. Pero el mismo «±» vale distinto
según dónde se lea; recomiendo unificarlo cuando el coordinador decida el ddof.

**N3 · Para el ingeniero de modelos: redondeo en el límite.** En la tabla `:36`, el RF muestra «Media
partición única − promedio CV» como −0.003. El JSON da −0.0025 (`media_particion_unica_menos_promedio_cv`),
que está justo en el límite de redondeo a 3 decimales. No cambia ninguna conclusión. Para que la tabla
y el JSON no parezcan discrepar, conviene escribir −0.0025.

**N4 · Para el ingeniero de modelos: eficiencia.** `experimentos.py:68-69` calcula
`cross_val_predict` dos veces con el mismo `cv` (una dentro de `proporcion_malos_detectados` y otra
para `pred`), y una tercera vez con `predict_proba` en `:73`. Los resultados son correctos y
deterministas. Solo cuesta tiempo (unos 33 s en total, dominado por las 20 semillas de H3).

## Veredicto de la ronda 2

H1–H5 **CERRADOS**. La reproducibilidad **PASA** (JSON idéntico byte a byte). Los puntos 1–5 de la
ronda 1 siguen en PASA, y el punto 6 (Interpretación) pasa a **PASA**. Las observaciones N1–N4 son
menores y no afectan a ningún número ni conclusión.
