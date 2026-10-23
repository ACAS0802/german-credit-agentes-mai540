# Riesgo de crédito con German Credit: una herramienta auditada

Estima la probabilidad de que una solicitud de crédito de consumo termine siendo un crédito malo, para que un analista de riesgo decida cuáles revisar a fondo antes de aprobar.

**Araceli Castillo** · MAI 540: Machine Learning · Prof. Kevin A. García Gallardo · Atlantis University · Proyecto Capstone, octubre de 2026

## Problema

Un analista de crédito recibe más solicitudes de las que puede revisar a fondo. La herramienta le dice cuáles tienen riesgo alto de terminar en impago, para que concentre la revisión en ellas. Los dos errores no cuestan lo mismo: aprobar un crédito que resulta malo cuesta **5 veces más** que rechazar uno bueno (la matriz de costos del conjunto original). Por eso el modelo se elige por **costo** y no solo por exactitud.

La herramienta **apoya** la decisión, no la toma: el resultado es «revisar a fondo» o «sin señales de riesgo alto», nunca «aprobar» o «rechazar».

## Datos

- **Fuente:** *Statlog (German Credit Data)*, Hofmann (1994), UCI Machine Learning Repository (también publicado en OpenML como `credit-g`). Copia en `datos/credit_g.csv`.
- **Permiso de uso:** licencia Creative Commons Attribution 4.0 (CC BY 4.0). Los datos están anonimizados: no identifican a ninguna persona.
- **Descripción:** 1,000 solicitudes de un banco alemán de los años 70, con 20 variables de la solicitud (estado de la cuenta corriente, duración y monto del crédito, historial, propósito, ahorro, empleo, vivienda, entre otras). La variable objetivo `clase` vale `bad` (crédito malo, 30 %) o `good` (70 %). Por eso un modelo que dijera siempre «bueno» acertaría el 70 %: esa es la línea base.
- **Variables sensibles:** la edad (`age`), el sexo y estado civil (`personal_status`) y la condición de trabajador extranjero (`foreign_worker`) **no entran al modelo**. Se usan solo para auditar si el modelo se equivoca más con algún grupo.

## Cómo funciona

```
solicitud (17 datos) ──► validación de la entrada ──► pipeline ──► probabilidad de crédito malo
                          herramienta.py              │
                                                      ├─ codificación de categorías (OrdinalEncoder)
                                                      └─ boosting con pesos balanceados (8 hojas por árbol)
```

| Pieza | Archivo | Qué hace |
|---|---|---|
| Datos | `datos.py` | Lee el CSV y define la validación cruzada estratificada de 5 pliegues (`random_state=42`) |
| Modelo | `modelos.py` | El pipeline (la codificación va dentro, para que no haya fuga entre pliegues) y el modelo elegido |
| Regla de elección | `regla.py` | Entre los modelos con exactitud mayor que 0.70, el de menor costo (5 × malo aprobado + 1 × bueno rechazado) |
| Selección anidada | `seleccion.py` | Aplica la regla dentro de cada pliegue, para que el modelo nunca se evalúe con los datos que lo eligieron |
| Auditoría | `auditoria.py` | Errores por subgrupo, con un umbral de disparidad justificado por el costo |
| Flujo completo | `capstone.py` | Compara siete modelos, elige con la regla, comprueba la robustez y audita al elegido |
| La herramienta | `herramienta.py` | Valida una solicitud y devuelve la probabilidad de crédito malo, con sus limitaciones |

## Resultados

Validación cruzada estratificada de 5 pliegues. Todos los modelos usan la misma partición y ninguno ve las variables sensibles. Las cifras de los modelos con búsqueda son **anidadas**: la búsqueda ocurre dentro de cada pliegue, sin ver su parte de prueba.

| Modelo | Exactitud (media ± desviación) | Malos detectados | Costo | Observación |
|---|---|---|---|---|
| Línea base: «siempre bueno» | 0.700 | 0 % | 1,500 | Aprueba todo |
| Árbol afinado | 0.714 ± 0.029 | 36 % | 1,058 | |
| Random Forest | 0.764 ± 0.026 | 41 % | 944 | La mejor exactitud, pero deja pasar casi 6 de cada 10 malos |
| Boosting balanceado (Tarea 6.2) | 0.752 ± 0.031 | 60 % | 728 | Memoriza: 0.996 en entrenamiento |
| **Boosting balanceado de menor capacidad (elegido)** | **0.704 ± 0.032** | **71 %** | **640** | Brecha entrenamiento-validación de 0.08 (antes 0.24) |
| Procedimiento completo de selección (anidado) | 0.711 ± 0.030 | 69 % | 661 | Lo que se espera de «elegir con la regla» con datos nuevos |

Las siete filas completas, con el árbol balanceado y las versiones balanceadas, están en `resultados/capstone.json` (`comparacion`).

**Por qué este modelo.** La regla lo elige porque es el de menor costo entre los que superan a «siempre bueno». Frente al de la 6.2 detecta 11 puntos más de créditos malos a cambio de rechazar más solicitudes buenas. Su exactitud apenas supera 0.70 (en 2 de 5 particiones alternativas queda en 0.695), así que **su valor está en el costo, no en la exactitud**: es el más barato en las 5 particiones (591 a 672, frente a 727 a 760 del de la 6.2). La capacidad se eligió explorando hojas por árbol, casos por hoja, tasa de aprendizaje y número de árboles, con entrenamiento y validación juntos (`resultados/capacidad_boosting.csv`, `figs/capacidad_boosting.png`): con 4 hojas el modelo se queda corto y con 31 memoriza.

**Errores por categoría** (`resultados/errores/`, predicciones anidadas):

| Dónde falla más | Malos detectados | Por qué |
|---|---|---|
| Sin cuenta corriente (`no checking`, 394 solicitudes, 46 malos) | 13 % | El modelo aprendió que no tener cuenta corriente es señal de buen pagador (solo 12 % de malos), y casi nunca marca a estos solicitantes |
| Créditos de hasta 12 meses (359 solicitudes, 76 malos) | 57 % | Los créditos cortos tienen pocos malos (21 %) y el modelo los trata como seguros |
| Montos del segundo cuartil (250 solicitudes, 62 malos) | 56 % | Montos medios y bajos parecen seguros para el modelo |
| Créditos de más de 36 meses | 93 % detectados, pero rechaza el 67 % de los buenos | Aquí falla al revés: casi todo lo marca como riesgo |

**Auditoría por subgrupo** (`resultados/subgrupos/`): ningún grupo de edad, sexo, vivienda u origen con al menos 30 créditos malos cae más de 0.10 por debajo del global (la mayor caída es 0.031, vivienda propia, sin diferencia significativa: p = 0.14). Los trabajadores no extranjeros (4 malos) y dos estados civiles de hombres (20 y 25 malos) son **no concluyentes**: hay muy pocos casos para saberlo. En los hombres casados o viudos la caída es 0.11, que con 25 casos no se distingue del azar (p = 0.19), pero merece vigilarse si llegan más datos.

## Instalación

Requiere Python 3.11 o superior (probado con 3.12 y 3.13) y Git. `requirements.txt` fija scikit-learn 1.8.0: con otra versión las cifras pueden cambiar en el tercer decimal (con 1.9.1 la probabilidad del ejemplo sale 0.788 en lugar de 0.792).

```bash
git clone https://github.com/ACAS0802/german-credit-agentes-mai540.git
cd german-credit-agentes-mai540
python3.12 -m venv .venv       # o python3.11 / python3.13
source .venv/bin/activate        # en Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python instalar_hook.py          # opcional: ejecuta las pruebas antes de cada commit
```

## Cómo ejecutar

**Evaluar una solicitud** (la herramienta):

```bash
python herramienta.py --ejemplo
```

Se debe ver algo así (tarda unos segundos, porque entrena el modelo al empezar):

```json
{
  "probabilidad_malo": 0.792,
  "riesgo": "alto",
  "recomendacion": "revisar a fondo antes de aprobar",
  "limitaciones": "Herramienta académica de apoyo, no de decisión. ..."
}
```

Para evaluar otra solicitud, guárdala en un JSON con los mismos campos que `EJEMPLO` en `herramienta.py` y ejecuta `python herramienta.py mi_solicitud.json`. Si falta un dato, si una categoría no existe o si un número está fuera de rango, la herramienta lo rechaza y dice qué corregir:

```bash
$ python herramienta.py mala.json
Solicitud inválida: 'duration' = -6 fuera de rango (1 a 120)
```

**Reproducir los resultados:**

```bash
python capstone.py               # unos 25 minutos: escribe resultados/capstone.json y las tablas
python capstone.py --auditoria   # solo la auditoría por subgrupo y categoría, en segundos
```

## Pruebas

```bash
pytest -v
```

Se espera `18 passed, 1 xfailed` en unos 5 segundos (el árbol sin límite de profundidad no supera la línea base; es la lección de la Tarea 6.1 y está marcado como fallo esperado). La salida guardada está en `resultados/pytest.txt`, y el registro de las tres comprobaciones antes de entregar (pruebas, hook y auditoría) en `resultados/comprobaciones.md`.

| Prueba | Qué comprueba |
|---|---|
| `test_supera_a_siempre_bueno`, `test_elegido_supera_a_siempre_bueno` | Los modelos superan la exactitud de «siempre bueno» (0.70) |
| `test_elegido_detecta_al_menos_la_mitad_de_los_malos` | El modelo detecta al menos la mitad de los créditos malos |
| `test_resultado_estable` | La desviación entre pliegues es menor que 0.05 (el muestreo solo explica unos 0.03) |
| `test_codificacion_dentro_del_pipeline` | La codificación está dentro del pipeline: no hay fuga entre pliegues |
| `test_variables_sensibles_no_deciden`, `test_edad_sexo_y_origen_no_cambian_la_respuesta` | Cambiar la edad, el sexo o el origen no cambia ninguna predicción |
| `test_modelo_elegido_es_el_que_eligio_la_regla` | Nadie cambió el modelo a mano: coincide con lo que eligió la regla |
| `test_regla_elige_el_menor_costo_entre_los_que_superan_070` | La regla no elige un modelo barato que no supera la línea base |
| `test_rechaza_entrada_invalida`, `test_rechaza_solicitud_incompleta` | La herramienta rechaza duraciones negativas, montos en cero, categorías inexistentes, texto en lugar de números y solicitudes incompletas |
| `test_devuelve_probabilidad_valida_y_limitaciones` | La respuesta trae una probabilidad entre 0 y 1 y el texto de limitaciones |
| `test_grupo_pequeno_es_no_concluyente_aunque_caiga_mucho`, `test_auditoria_detecta_una_disparidad_real` | La auditoría no aprueba grupos pequeños y sigue detectando una disparidad real |

## Limitaciones

- **No debe usarse para decidir sobre personas reales.** Se entrenó con 1,000 solicitudes alemanas de hace 50 años. Deja pasar cerca de 3 de cada 10 créditos malos y rechaza cerca de 3 de cada 10 buenos.
- **No sirve para solicitantes sin cuenta corriente ni para créditos cortos**: ahí detecta el 13 % y el 57 % de los malos.
- **Excluir las variables sensibles no las vuelve invisibles.** Las demás variables permiten adivinar en parte la edad, el sexo y el origen (AUC entre 0.69 y 0.81). El modelo no las usa directamente, pero sí puede usar sus huellas.
- **Su ventaja en exactitud sobre «siempre bueno» es mínima** (0.704 frente a 0.700). Lo que lo justifica es el costo.
- **Qué datos cambiarían con el tiempo:** la duración típica de los créditos, los montos, el perfil de quienes piden crédito y el uso de cuentas corrientes. Si cambian, los aciertos bajan sin que nada avise.
- **Cómo se notaría:** midiendo cada mes los malos detectados y el costo con créditos nuevos cuyo resultado ya se conoce. Hoy detecta 0.71 de los malos y cuesta 0.64 por solicitud. Si los malos detectados bajan de 0.60 o el costo por solicitud sube de 0.73 (lo que costaba el modelo de la 6.2), hay que revisarlo.
- **Cuándo revisar o reentrenar:** cada vez que cambie el producto o la población que pide crédito, y al menos una vez al año.

## Estructura del repositorio

```
.
├── README.md                 este archivo
├── requirements.txt
├── datos/credit_g.csv        German Credit (1,000 solicitudes)
├── datos.py, modelos.py, regla.py, seleccion.py, auditoria.py, busqueda.py
├── capstone.py               flujo final del capstone (Tarea 8.1)
├── herramienta.py            la herramienta: evalúa una solicitud
├── hooks/pre-commit, instalar_hook.py   pruebas antes de cada commit (Módulo 7)
├── tests/                    pruebas automáticas (pytest)
├── resultados/               capstone.json, capacidad, subgrupos, errores, pytest.txt
├── figs/                     figuras
├── auditoria/                informes de la Skill: ronda 1, ronda 2 y ronda 3
├── .claude/                  agentes (Tarea 6.1) y la Skill auditoria-modelos v5
└── historial/                README de las Tareas 6.1 y 6.2
```

Los archivos de las Tareas 6.1 y 6.2 (`main.py`, `experimentos.py`, `optimizacion.py`, `validacion.py`, `curvas.py`, `entregas/`, `evidencia/` y los informes en PDF) se conservan como historial del proyecto. `main.py` es el flujo de la clase y usa todas las variables; los resultados del capstone salen de `capstone.py`.

## Autora

Araceli Castillo · MAI 540: Machine Learning · Atlantis University · octubre de 2026
