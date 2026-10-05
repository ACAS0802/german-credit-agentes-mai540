# Demo Clase 6 · Evaluación de modelos con German Credit

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

