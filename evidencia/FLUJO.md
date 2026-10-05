# Evidencia del flujo distribuido

Cada rol se ejecutó como un **agente separado** (Claude Code, subagente de propósito general) que recibió
como instrucciones solo su archivo de `.claude/agents/` y los entregables del rol anterior. Ningún rol
vio la conversación de los otros: la única comunicación fue a través de los archivos de `entregas/`.
Cada entrega quedó en su propio commit (`git log --oneline`).

## Cadena de entregas

| Paso | Rol | Recibe | Entrega | Commit |
|---|---|---|---|---|
| 1 | Analista de datos | `datos/credit_g.csv` | `datos.py` (TODO 1), `01_analista_resumen.md`, `01_analista_datos.json` | `7b889d2` |
| 2 | Ingeniero de modelos | Entregas 01 | `modelos.py`, `validacion.py` (TODO 2-6), `experimentos.py`, `02_ingeniero_metricas.json`, `02_ingeniero_reporte.md` | `b0ef253` |
| 3 | Auditor | Entregas 01 y 02 + código | `03_auditor_reporte.md`: 5 PASA, 1 FALLA, hallazgos H1-H5 | `08e7442` |
| 4a | Analista (corrección) | H4 | `01_analista_resumen.md` corregido | `eebc721` |
| 4b | Ingeniero (corrección) | H1, H2, H3, H5 | `experimentos.py`, entregas 02 con «Respuesta a la auditoría» | `1263731` |
| 5 | Auditor (ronda 2) | Correcciones | `03_auditor_reporte_ronda2.md`: H1-H5 cerrados | `a6d0589` |

## Lo que detectó el auditor y ningún otro rol vio

- **H1:** el ingeniero comparaba el **rango** de 10 particiones (0.112) con la **desviación** de 5 pliegues
  (0.017). Comparando lo mismo con lo mismo: rango 0.112 frente a 0.045, o desviación 0.031 frente a 0.019.
- **H3:** «el árbol no supera 0.70» se apoyaba en una sola semilla de la validación cruzada. Con 20 semillas,
  el árbol supera 0.70 solo 2 de 20 veces (media 0.677).
- **H4:** el analista escribió que un modelo con exactitud ≤ 0.70 «no aporta nada», pero el árbol (0.698)
  detecta 53 % de los créditos malos.
- **H2:** que el Random Forest detecte menos malos que el árbol solo es cierto con el umbral 0.5; con 0.4
  detecta 0.603 y supera al árbol en exactitud y en detección.

## Costo medido (primera ronda; lo reporta la herramienta al terminar cada agente)

| Agente | Tokens | Tiempo | Llamadas a herramientas |
|---|---|---|---|
| Analista | 79,870 | 52 s | 9 |
| Ingeniero | 82,546 | 52 s | 4 |
| Auditor | 97,597 | 186 s | 9 |
| **Total flujo (ronda 1)** | **260,013** | **290 s** | 22 |
| Un solo agente (línea base) | 79,809 | 77 s | 6 |

La ronda 2 (correcciones y nueva auditoría) no reportó su costo y no está incluida.

## Línea base: un solo agente

Para comparar, un agente aparte hizo toda la tarea en una sola conversación sobre una copia limpia del
proyecto base (`linea_base_un_agente/`). Obtuvo **exactamente los mismos números** (árbol 0.698 ± 0.017,
detecta 0.530; RF 0.759 ± 0.006, detecta 0.377; particiones 0.608-0.720 y 0.712-0.796) y su
autoverificación dio «Correcto» en fuga, etiqueta y reproducibilidad. Pero su texto repite los problemas
que el auditor sí marcó en el flujo distribuido: afirma que el árbol «no supera la referencia» con 0.698
frente a 0.700 sin revisar la semilla (H3), contrasta una desviación de 0.006 con «±4 puntos» de las
particiones (H1) y no dice que la comparación del recall depende del umbral (H2). Su autoverificación
solo revisó lo que ya sabía revisar.
