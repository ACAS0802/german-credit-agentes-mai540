"""Ejecuta todo el flujo: datos → modelos → validación → curva → búsqueda → auditoría."""
import os
from datos import cargar_datos
from modelos import crear_pipeline, tres_modelos
from validacion import validar, proporcion_malos_detectados, particion_unica
from curvas import curva_aprendizaje, graficar
from busqueda import buscar, buscar_con_prueba
from auditoria import exactitud_por_subgrupo

X, y, cols_cat, cv = cargar_datos()                        # paso 1: datos listos
os.makedirs("figs", exist_ok=True)                         # carpeta para las gráficas

print("\n== 1. Tres modelos con la misma validación cruzada ==")
for nombre, modelo in tres_modelos().items():
    pipe = crear_pipeline(modelo, cols_cat)                # cada modelo con su pipeline
    media, desv = validar(pipe, X, y, cv)                  # media y desviación de la exactitud
    malos = proporcion_malos_detectados(pipe, X, y, cv)    # qué proporción de créditos malos detecta
    print(f"{nombre:14s} exactitud {media:.3f} ± {desv:.3f}   detecta {malos:.2f} de los malos")

print("\n== 2. Una sola partición frente a la validación cruzada (árbol) ==")
arbol = crear_pipeline(tres_modelos()["Árbol"], cols_cat)
accs = particion_unica(arbol, X, y)                        # 10 particiones distintas
print(f"una partición: de {accs.min():.3f} a {accs.max():.3f}  | validación cruzada: {validar(arbol, X, y, cv)[0]:.3f}")

print("\n== 3. Curvas de aprendizaje (se guardan en figs/) ==")
for nombre in ("Árbol", "Random Forest"):
    pipe = crear_pipeline(tres_modelos()[nombre], cols_cat)
    t, ent, val = curva_aprendizaje(pipe, X, y, cv)
    graficar(t, ent, val, nombre, f"figs/curva_{nombre.replace(' ', '_')}.png")
    print(f"{nombre}: entrenamiento {ent[-1]:.2f} frente a validación {val[-1]:.2f} (brecha {ent[-1]-val[-1]:.2f})")

print("\n== 4. Búsqueda de hiperparámetros del árbol ==")
mejor, puntaje = buscar(arbol, X, y, cv)
print("mejor combinación:", mejor, "| exactitud en la búsqueda:", round(puntaje, 3))
busq, prueba = buscar_con_prueba(arbol, X, y, cv)
print(f"con prueba reservada: la búsqueda dijo {busq:.3f} y la prueba {prueba:.3f}")

print("\n== 5. Auditoría del Random Forest por subgrupo ==")
rf = crear_pipeline(tres_modelos()["Random Forest"], cols_cat)
for columna in ("housing", "personal_status"):
    print(f"\n{columna}"); print(exactitud_por_subgrupo(rf, X, y, cv, columna).round(2))
