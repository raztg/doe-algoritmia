import itertools
import math
import random
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Generación de Instancias TSP para n = 5, 10, 15
# -------------------------------------------------------------
def generate_cities(n, seed=42):
    random.seed(seed)
    cities = {}
    names = [chr(65 + i) for i in range(n)]
    for name in names:
        cities[name] = (random.randint(0, 20), random.randint(0, 20))
    return cities

def tour_length(route: list[str], cities: dict) -> float:
    return sum(math.dist(cities[route[i]], cities[route[(i + 1) % len(route)]])
               for i in range(len(route)))

def two_opt_move_random(route: list[str]) -> list[str]:
    i, k = random.sample(range(len(route)), 2)
    if i > k:
        i, k = k, i
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]

def roll(n: float) -> bool:
    if n == 0:
        return False
    return random.random() < 1 / n

def simulated_annealing(route: list[str], T_init: float, cities: dict, max_iter=1000):
    current = list(route)
    T = T_init
    t = 0
    while t < max_iter and T > 0:
        t += 1
        T = T * 0.95
        new = two_opt_move_random(current)
        delta_E = tour_length(new, cities) - tour_length(current, cities)
        if delta_E <= 0:
            current = list(new)
        else:
            if roll(math.exp(-1 * delta_E / T)):
                current = list(new)
    return current, tour_length(current, cities)

# -------------------------------------------------------------
# 2. Configuración del DOE y Temperaturas
# -------------------------------------------------------------
r_val = random.uniform(0, 1)
print(f"Valor aleatorio r fijado: {r_val:.4f}")

T_min, T_max = 100, 2000
frac_bajo = 0.3 * r_val
frac_medio = 0.3 * r_val + 0.3
frac_alto = 0.3 * r_val + 0.6

temp_bajo = T_min + frac_bajo * (T_max - T_min)
temp_medio = T_min + frac_medio * (T_max - T_min)
temp_alto = T_min + frac_alto * (T_max - T_min)

temperaturas = {
    "Bajo": temp_bajo,
    "Medio": temp_medio,
    "Alto": temp_alto
}
print(f"Niveles de Temperatura: {temperaturas}\n")

tamanios = [5, 10, 15]
replicas = 10

data = []

# -------------------------------------------------------------
# 3. Ejecución del Experimento (90 corridas)
# -------------------------------------------------------------
run_id = 0
for n in tamanios:
    cities = generate_cities(n, seed=100 + n)
    initial_route = list(cities.keys())

    for temp_name, T_val in temperaturas.items():
        for rep in range(replicas):
            run_id += 1
            random.seed(run_id * 37)
            _, final_cost = simulated_annealing(initial_route, T_val, cities)

            data.append({
                "run": run_id,
                "tamano": n,
                "temperatura": temp_name,
                "T_val": T_val,
                "replica": rep + 1,
                "costo": final_cost
            })

df = pd.DataFrame(data)

# -------------------------------------------------------------
# 4. ANOVA de Dos Factores
# -------------------------------------------------------------
modelo = ols('costo ~ C(temperatura) + C(tamano) + C(temperatura):C(tamano)', data=df).fit()
tabla_anova = sm.stats.anova_lm(modelo, typ=2)
print("--- TABLA ANOVA DE DOS FACTORES ---")
print(tabla_anova)
print("\n")

# -------------------------------------------------------------
# 5. Bootstrap para la Mediana e Intervalos de Confianza (95%)
# -------------------------------------------------------------
def bootstrap_median_ci(series, num_bootstrap=1000, alpha=0.05):
    boot_medians = []
    n_samples = len(series)
    for _ in range(num_bootstrap):
        sample = np.random.choice(series, size=n_samples, replace=True)
        boot_medians.append(np.median(sample))
    lower = np.percentile(boot_medians, 100 * (alpha / 2))
    upper = np.percentile(boot_medians, 100 * (1 - alpha / 2))
    return np.median(series), lower, upper

print("--- MEDIANA Y BOOTSTRAP CI (95%) POR PUNTO DE DISEÑO ---")
summary_results = []
for n in tamanios:
    for temp_name in temperaturas.keys():
        subset = df[(df['tamano'] == n) & (df['temperatura'] == temp_name)]['costo'].values
        med, low, high = bootstrap_median_ci(subset)
        summary_results.append({
            "Tamaño (n)": n,
            "Temperatura": temp_name,
            "Mediana Costo": med,
            "CI_Inferior": low,
            "CI_Superior": high
        })

df_summary = pd.DataFrame(summary_results)
print(df_summary.to_string(index=False))

# -------------------------------------------------------------
# 6. Visualizaciones (Gráficas de Efectos e Interacción)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfica de Efectos Principales / Interacción
interaction_data = df.groupby(['tamano', 'temperatura'])['costo'].mean().unstack()
interaction_data.plot(kind='bar', ax=axes[0], colormap='viridis', edgecolor='black')
axes[0].set_title("Interacción: Tamaño vs Temperatura (Costo Promedio)")
axes[0].set_xlabel("Tamaño del Problema (n)")
axes[0].set_ylabel("Costo Promedio de la Ruta")
axes[0].legend(title="Temperatura")
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Efecto Principal Temperatura
temp_effect = df.groupby('temperatura')['costo'].mean().reindex(['Bajo', 'Medio', 'Alto'])
temp_effect.plot(kind='line', marker='o', ax=axes[1], color='b', linewidth=2, markersize=8)
axes[1].set_title("Efecto Principal: Temperatura")
axes[1].set_xlabel("Nivel de Temperatura")
axes[1].set_ylabel("Costo Promedio")
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("doe_analisis_temperatura.png", dpi=300)
print("\nGráfica guardada exitosamente como 'doe_analisis_temperatura.png'.")
