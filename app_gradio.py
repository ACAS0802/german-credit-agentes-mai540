"""Demostración ejecutable (Tarea 8.2): la herramienta con una interfaz web de Gradio.

Uso local:   python app_gradio.py      y abre la dirección que aparece (http://127.0.0.1:7860)
Uso en Colab: demo_colab.ipynb (crea un enlace público temporal con share=True)

La interfaz no tiene modelo propio: llama a evaluar_solicitud() de herramienta.py, la misma función que
cubren las pruebas. Así la demo usa exactamente el modelo que eligió la regla y la misma validación de
entradas. Basada en ejemplo_despliegue/app_gradio.py del Kit del Módulo 8.
"""
import sys

import gradio as gr

from herramienta import EJEMPLO, LIMITACIONES, _modelo_y_esquema, evaluar_solicitud

# Etiquetas en español para cada variable de la solicitud (el modelo usa los nombres en inglés del dataset).
ETIQUETAS = {
    "checking_status": "Saldo de la cuenta corriente (marcos alemanes)",
    "duration": "Duración del crédito (meses)",
    "credit_history": "Historial de crédito",
    "purpose": "Propósito del crédito",
    "credit_amount": "Monto solicitado (marcos alemanes)",
    "savings_status": "Ahorros (marcos alemanes)",
    "employment": "Años en el empleo actual",
    "installment_commitment": "Cuota como % del ingreso (1 = menos de 20 %, 4 = 35 % o más)",
    "other_parties": "Codeudor o garante",
    "residence_since": "Años en la vivienda actual (1 a 4)",
    "property_magnitude": "Bien de mayor valor",
    "other_payment_plans": "Otros planes de pago",
    "housing": "Vivienda",
    "existing_credits": "Créditos vigentes en el banco",
    "job": "Tipo de empleo",
    "num_dependents": "Personas a cargo",
    "own_telephone": "Teléfono a su nombre",
}
NUMERICAS = {
    "duration": (4, 72, 1), "credit_amount": (250, 20000, 50), "installment_commitment": (1, 4, 1),
    "residence_since": (1, 4, 1), "existing_credits": (1, 4, 1), "num_dependents": (0, 4, 1),
}
# Traducción de las categorías para quien usa la app. El valor que recibe el modelo no cambia.
TRADUCCION = {
    "0<=X<200": "entre 0 y 200", "<0": "negativo (sobregiro)", ">=200": "200 o más", "no checking": "no tiene cuenta corriente",
    "all paid": "todos los créditos de este banco pagados", "critical/other existing credit": "cuenta crítica / créditos en otros bancos",
    "delayed previously": "pagos atrasados en el pasado", "existing paid": "créditos vigentes al día",
    "no credits/all paid": "sin créditos o todos pagados",
    "business": "negocio", "domestic appliance": "electrodomésticos", "education": "educación",
    "furniture/equipment": "muebles / equipo", "new car": "auto nuevo", "other": "otro", "radio/tv": "radio / TV",
    "repairs": "reparaciones", "retraining": "capacitación", "used car": "auto usado",
    "100<=X<500": "entre 100 y 500", "500<=X<1000": "entre 500 y 1,000", "<100": "menos de 100", ">=1000": "1,000 o más",
    "no known savings": "sin ahorros conocidos",
    "1<=X<4": "de 1 a 4 años", "4<=X<7": "de 4 a 7 años", "<1": "menos de 1 año", ">=7": "7 años o más", "unemployed": "desempleado",
    "co applicant": "cosolicitante", "guarantor": "garante", "none": "ninguno",
    "car": "auto", "life insurance": "seguro de vida", "no known property": "sin bienes conocidos", "real estate": "inmueble",
    "bank": "con un banco", "stores": "con tiendas",
    "for free": "sin pagar (familiar, empleador)", "own": "propia", "rent": "alquilada",
    "high qualif/self emp/mgmt": "alta calificación / independiente / gerente", "skilled": "empleado calificado",
    "unemp/unskilled non res": "desempleado / no calificado no residente", "unskilled resident": "no calificado residente",
    "yes": "sí",
}
CAMPOS = list(ETIQUETAS)                      # el orden de los controles en la pantalla

# Una solicitud de riesgo bajo, para comparar con EJEMPLO (riesgo alto)
EJEMPLO_BAJO = {**EJEMPLO, "checking_status": "no checking", "duration": 12, "credit_history": "critical/other existing credit",
                "purpose": "radio/tv", "credit_amount": 1500, "savings_status": ">=1000", "employment": ">=7",
                "installment_commitment": 2, "property_magnitude": "real estate", "housing": "own"}


def evaluar(*valores):
    """Recibe los valores de la pantalla en el orden de CAMPOS y devuelve el resultado en Markdown."""
    solicitud = {c: (int(v) if c in NUMERICAS else v) for c, v in zip(CAMPOS, valores)}
    try:
        r = evaluar_solicitud(solicitud)
    except ValueError as e:
        return f"### ⚠️ Solicitud no válida\n{e}"
    icono = "🔴" if r["riesgo"] == "alto" else "🟢"
    return (f"### {icono} Riesgo {r['riesgo']}: probabilidad de crédito malo {r['probabilidad_malo']:.0%}\n"
            f"**Recomendación:** {r['recomendacion']}.\n\n"
            "_El umbral es el del modelo con pesos balanceados: marca «alto» cuando estima que ser malo es "
            "más probable que ser bueno, después de compensar que los malos son solo el 30 % de los datos._")


def construir():
    _, _, categorias = _modelo_y_esquema()
    entradas = []
    for c in CAMPOS:
        if c in NUMERICAS:
            mn, mx, paso = NUMERICAS[c]
            entradas.append(gr.Slider(mn, mx, value=EJEMPLO[c], step=paso, label=ETIQUETAS[c]))
        else:
            entradas.append(gr.Dropdown([(TRADUCCION.get(v, v), v) for v in categorias[c]], value=EJEMPLO[c], label=ETIQUETAS[c]))
    return gr.Interface(
        fn=evaluar,
        inputs=entradas,
        outputs=gr.Markdown(label="Resultado"),
        examples=[[EJEMPLO[c] for c in CAMPOS], [EJEMPLO_BAJO[c] for c in CAMPOS]],
        example_labels=["Solicitud de riesgo alto", "Solicitud de riesgo bajo"],
        title="Riesgo de crédito · German Credit (MAI 540, Araceli Castillo)",
        description=(f"**Para qué sirve:** ayudar a un analista a decidir qué solicitudes revisar a fondo antes de aprobar.\n\n"
                     f"**⚠️ Limitaciones (léelas antes de usarla):** {LIMITACIONES}\n\n"
                     "La edad, el sexo, el estado civil y el origen no se piden: el modelo no los usa."),
        flagging_mode="never",
        submit_btn="Evaluar",
        clear_btn=None,
    )


if __name__ == "__main__":
    construir().launch(share="--share" in sys.argv)
