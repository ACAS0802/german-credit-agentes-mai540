"""Resultados: árbol y Random Forest con validación cruzada frente a una sola partición 75/25."""
from datos import cargar_datos
from modelos import crear_pipeline, tres_modelos
from validacion import validar, proporcion_malos_detectados, particion_unica

X, y, cols_cat, cv = cargar_datos()
print(f"Casos: {len(y)} | proporción de malos (positivo=1): {y.mean():.3f}\n")
print(f"{'Modelo':14s} {'CV media':>8s} {'CV desv':>8s} {'malos det.':>10s} "
      f"{'part. mín':>9s} {'part. máx':>9s} {'rango':>6s} {'mín-CV':>7s} {'máx-CV':>7s}")
for nombre in ("Árbol", "Random Forest"):
    pipe = crear_pipeline(tres_modelos()[nombre], cols_cat)
    media, desv = validar(pipe, X, y, cv)
    malos = proporcion_malos_detectados(pipe, X, y, cv)
    accs = particion_unica(pipe, X, y, semillas=range(10))
    print(f"{nombre:14s} {media:8.3f} {desv:8.3f} {malos:10.3f} {accs.min():9.3f} {accs.max():9.3f} "
          f"{accs.max()-accs.min():6.3f} {accs.min()-media:+7.3f} {accs.max()-media:+7.3f}")
    print(f"{'':14s} particiones (semillas 0-9): {[round(float(a), 3) for a in accs]} | media {accs.mean():.3f}")
