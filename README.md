# Flujo de Trabajo con Agentes Especializados — Tarea 6.1 (MAI 540)

**Araceli Castillo** · MAI 540: Machine Learning · Prof. Kevin A. García Gallardo · Atlantis University · Octubre de 2026

Evaluación de un árbol de decisión y un Random Forest sobre German Credit (1,000 solicitudes, 30 % «bad»),
repartiendo el trabajo entre tres roles que se ejecutaron como agentes separados.

| Dónde | Qué |
|---|---|
| `.claude/agents/` | Los tres roles: `analista-datos.md`, `ingeniero-modelos.md`, `auditor.md` |
| `entregas/` | Lo que entregó cada rol, en orden (01 analista, 02 ingeniero, 03 auditor y su ronda 2) |
| `evidencia/FLUJO.md` | Cadena de entregas con su commit, hallazgos, costo de cada agente y línea base de un solo agente |
| `experimentos.py` | Validación cruzada, partición única y comprobaciones pedidas por el auditor |
| `MAI540_Tarea6.1_Informe_Araceli_Castillo.pdf` | Informe APA |

**Reproducir** (Python 3.10+, semillas documentadas):

```bash
pip install -r requirements.txt
python datos.py          # forma de X, proporción de malos, columnas categóricas
python experimentos.py   # escribe entregas/02_ingeniero_metricas.json
```

**Resultados** (validación cruzada estratificada de 5 pliegues, `random_state=42`):

| Modelo | Exactitud (media ± desv., ddof=1) | Malos detectados | Partición única 75/25, semillas 0-9 |
|---|---|---|---|
| Árbol de decisión | 0.698 ± 0.019 | 0.530 | 0.608 – 0.720 (rango 0.112) |
| Random Forest | 0.759 ± 0.007 | 0.377 | 0.712 – 0.796 (rango 0.084) |

`main.py` aún no corre completo: los TODO 7 a 12 (curvas, búsqueda, auditoría por subgrupo y pruebas)
quedan para la Tarea 6.2, que parte de este diagnóstico.

---

*Lo que sigue es el README original del proyecto base del profesor.*

## Demo Clase 6 · Evaluación de modelos con German Credit

Proyecto que se programa en vivo en la Clase 6. Evalúa tres modelos (árbol, Random Forest y boosting)
sobre 1,000 solicitudes de crédito: validación cruzada, curva de aprendizaje, búsqueda de hiperparámetros,
prueba reservada, auditoría por subgrupos y pruebas automáticas. Esta es la versión **incompleta**: los bloques marcados con `TODO` se escriben en clase.

## Instalación (gratis, sin claves de API)

Requiere Python 3.10 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Qué hay en cada archivo

| Archivo | Qué hace |
|---|---|
| `datos.py` | Lee el CSV y deja listos X, y, columnas categóricas y la validación cruzada |
| `modelos.py` | El Pipeline (codificación dentro) y los tres modelos |
| `validacion.py` | Validación cruzada, créditos malos detectados y partición única |
| `curvas.py` | Curva de aprendizaje |
| `busqueda.py` | GridSearchCV y la prueba reservada |
| `auditoria.py` | Exactitud por subgrupo |
| `main.py` | Ejecuta todo el flujo y muestra los resultados |
| `tests/test_modelo.py` | Pruebas automáticas con pytest |
| `.claude/agents/` | Definición de los agentes especializados |

## Cómo se completa

Cada `TODO n` marca un bloque que se escribe en vivo. Mientras un bloque esté vacío, la función lanza
`NotImplementedError`. Para verificar cada paso por separado:

```bash
python datos.py        # después del TODO 1
python main.py         # cuando estén todos los bloques
pytest -v              # al final, con los TODO 11 y 12
```
La versión terminada de referencia está en `Demo_Credito_completo/`.

