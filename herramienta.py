"""La herramienta: evalúa una solicitud de crédito y devuelve el riesgo de que sea un crédito malo.

Uso desde la terminal:
    python herramienta.py --ejemplo                 # evalúa una solicitud de ejemplo
    python herramienta.py solicitud.json            # evalúa la solicitud guardada en un JSON

Uso desde Python:
    from herramienta import evaluar_solicitud
    evaluar_solicitud({"checking_status": "<0", "duration": 24, ...})

El modelo es modelo_elegido() de modelos.py, entrenado con los 1,000 casos de datos/credit_g.csv al
primer uso (tarda menos de un segundo). Antes de predecir, la solicitud se valida: si falta un dato,
si una categoría no existe en los datos o si un número está fuera de rango, se rechaza con un
mensaje que dice qué corregir. La edad, el estado civil/sexo y el trabajador extranjero se pueden
enviar, pero el modelo no los usa.
"""
import json
import sys
from functools import lru_cache

import pandas as pd

from datos import cargar_datos
from modelos import VARIABLES_SENSIBLES, crear_pipeline, modelo_elegido

LIMITACIONES = (
    "Herramienta académica de apoyo, no de decisión. Se entrenó con 1,000 solicitudes alemanas de los "
    "años 70 (German Credit); acierta en torno al 70 %, deja pasar cerca de 3 de cada 10 créditos malos y "
    "rechaza cerca de 3 de cada 10 buenos. Casi no detecta riesgo en solicitantes sin cuenta corriente. "
    "No debe usarse para aprobar o rechazar a personas reales. Si cambian el producto o el perfil de "
    "quienes piden crédito, sus aciertos bajan sin avisar."
)

# Rangos de los números. Mínimos y máximos lógicos del formulario de solicitud (no los del conjunto),
# para aceptar casos nuevos razonables y rechazar los imposibles.
RANGOS = {
    "duration": (1, 120),                 # meses
    "credit_amount": (1, 1_000_000),      # marcos alemanes
    "installment_commitment": (1, 4),     # cuota como % del ingreso, en 4 tramos
    "residence_since": (1, 4),            # años en la vivienda, en 4 tramos
    "existing_credits": (1, 10),          # créditos vigentes en el banco, contando este
    "num_dependents": (0, 10),            # personas a cargo
    "age": (18, 100),                     # solo se valida; el modelo no la usa
}


@lru_cache(maxsize=1)
def _modelo_y_esquema():
    """Entrena el modelo una sola vez y guarda qué categorías y columnas acepta."""
    X, y, cols_cat, _ = cargar_datos()
    modelo = crear_pipeline(modelo_elegido(), cols_cat, VARIABLES_SENSIBLES).fit(X, y)
    categorias = {c: sorted(X[c].unique()) for c in cols_cat}
    return modelo, list(X.columns), categorias


def validar_solicitud(solicitud):
    """Devuelve la solicitud como tabla de una fila, o lanza ValueError con lo que hay que corregir."""
    if not isinstance(solicitud, dict):
        raise ValueError("La solicitud debe ser un diccionario {variable: valor}.")
    _, columnas, categorias = _modelo_y_esquema()
    errores = []
    usadas = [c for c in columnas if c not in VARIABLES_SENSIBLES]
    faltan = [c for c in usadas if c not in solicitud]
    if faltan:
        errores.append(f"faltan datos: {', '.join(faltan)}")
    sobran = [c for c in solicitud if c not in columnas]
    if sobran:
        errores.append(f"variables desconocidas: {', '.join(sobran)}")
    for c, v in solicitud.items():
        if c in categorias and v not in categorias[c]:
            errores.append(f"'{c}' = {v!r} no es válido; opciones: {categorias[c]}")
        elif c in RANGOS:
            minimo, maximo = RANGOS[c]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                errores.append(f"'{c}' debe ser un número")
            elif not minimo <= v <= maximo:
                errores.append(f"'{c}' = {v} fuera de rango ({minimo} a {maximo})")
    if errores:
        raise ValueError("Solicitud inválida: " + "; ".join(errores))
    fila = {c: solicitud.get(c, categorias[c][0] if c in categorias else 0) for c in columnas}
    return pd.DataFrame([fila], columns=columnas)


def evaluar_solicitud(solicitud):
    """Probabilidad de crédito malo y recomendación. Valida la solicitud antes de predecir."""
    modelo, _, _ = _modelo_y_esquema()
    fila = validar_solicitud(solicitud)
    prob = float(modelo.predict_proba(fila)[0, 1])
    alto = bool(modelo.predict(fila)[0] == 1)
    return {
        "probabilidad_malo": round(prob, 3),
        "riesgo": "alto" if alto else "bajo",
        "recomendacion": ("revisar a fondo antes de aprobar" if alto else "sin señales de riesgo alto"),
        "limitaciones": LIMITACIONES,
    }


EJEMPLO = {
    "checking_status": "<0", "duration": 36, "credit_history": "existing paid", "purpose": "new car",
    "credit_amount": 6000, "savings_status": "<100", "employment": "1<=X<4", "installment_commitment": 4,
    "other_parties": "none", "residence_since": 2, "property_magnitude": "car", "other_payment_plans": "none",
    "housing": "rent", "existing_credits": 1, "job": "skilled", "num_dependents": 1, "own_telephone": "none",
}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python herramienta.py --ejemplo   o   python herramienta.py solicitud.json")
    if sys.argv[1] == "--ejemplo":
        datos = EJEMPLO
    else:
        with open(sys.argv[1], encoding="utf-8") as f:
            datos = json.load(f)
    try:
        resultado = evaluar_solicitud(datos)
    except ValueError as e:
        sys.exit(str(e))
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
