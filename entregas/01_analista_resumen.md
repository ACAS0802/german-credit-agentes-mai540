# 01 · Analista de datos: resumen de German Credit

Todos los números salen de ejecutar `cargar_datos()` sobre `datos/credit_g.csv`
(versión en máquina: `01_analista_datos.json`).

## Forma y objetivo
- `X`: **1000 filas × 20 columnas**; `y` = 1 si `clase == "bad"`.
- Malos: **300 (0.30)**; buenos: 700 (0.70).
- **Referencia «siempre bueno»: exactitud 0.70** y detecta 0.00 de los malos. Un modelo con exactitud ≤ 0.70 no mejora la exactitud de la referencia; hay que mirar también la proporción de malos detectados.

## Columnas
- **13 categóricas** (texto): checking_status (4 cat.), credit_history (5), purpose (10), savings_status (5),
  employment (5), personal_status (4), other_parties (3), property_magnitude (4), other_payment_plans (3),
  housing (3), job (4), own_telephone (2), foreign_worker (2).
- **7 numéricas** (enteras): duration (4–72), credit_amount (250–18424, media 3271), installment_commitment (1–4),
  residence_since (1–4), age (19–75), existing_credits (1–4), num_dependents (1–2).
- Faltantes: **0**. Filas duplicadas: **0**.

## Validación cruzada (fija, no cambiar sin avisar)
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`: 5 pliegues de 800 entrenamiento / 200 validación,
cada uno con proporción de malos 0.30.

## Advertencias para el ingeniero de modelos
1. **Desbalance 70/30**: reportar también la proporción de malos detectados, no solo exactitud.
2. **Codificar dentro del Pipeline**: las 13 categóricas son texto; el codificador debe ajustarse en cada pliegue.
3. **Categorías raras**: purpose «retraining» (n=9), «domestic appliance» y «other» (n=12 cada una),
   job «unemp/unskilled non res» (n=22). En particiones pequeñas pueden no aparecer en entrenamiento:
   usar `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`.
4. **Orden de OrdinalEncoder**: es alfabético, no semántico (p. ej. checking_status `<0`, `0<=X<200`, `>=200`,
   `no checking`). Sirve para árboles; no interpretar el código como magnitud.
5. **Subgrupos para la auditoría** (proporción de malos, n):
   - housing: own 0.261 (713), rent 0.391 (179), for free 0.407 (108).
   - personal_status: male single 0.266 (548), female div/dep/mar 0.352 (310), male mar/wid 0.272 (92), male div/sep 0.400 (50).
   - foreign_worker: yes 0.307 (963), no 0.108 (37). Grupos de n < 100 darán exactitudes inestables.

---
Corrección (hallazgo H4 del auditor): cambié «no aporta nada» por «no mejora la exactitud de la referencia», porque un modelo con exactitud ≤ 0.70 puede detectar malos (el árbol, 0.698, detecta el 53 %).
