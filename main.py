import itertools
import math
import multiprocessing as mp
import random
import time

log_results = False

##################################################
print("LOG RESULTS", log_results)
input("BEGIN simulated_annealing ...\n")
##################################################

if log_results: log1 = open("log4-1.csv", "w", encoding = "utf-8")
if log_results: log2 = open("log4-2.csv", "w", encoding = "utf-8")
if log_results: log3 = open("log4-3.csv", "w", encoding = "utf-8")
if log_results: log4 = open("log4-4.csv", "w", encoding = "utf-8")
if log_results: log5 = open("log4-5.csv", "w", encoding = "utf-8")
if log_results: log6 = open("log4-6.csv", "w", encoding = "utf-8")
if log_results: log7 = open("log4-7.csv", "w", encoding = "utf-8")
if log_results: log8 = open("log4-8.csv", "w", encoding = "utf-8")
if log_results: log1.write("Cost,Iterations\n")
if log_results: log2.write("Cost,Iterations\n")
if log_results: log3.write("Cost,Iterations\n")
if log_results: log4.write("Cost,Iterations\n")
if log_results: log5.write("Time,Quality\n")
if log_results: log6.write("Time,Quality\n")
if log_results: log7.write("Time,Quality\n")
if log_results: log8.write("Time,Quality\n")

cities = {"A": (0, 0), "B": (0, 3), "C": (2, 5), "D": (4, 3), "E": (4, 0), "F": (0, 5), "G": (1, 5), "H": (5, 5)}

def tour_length(route: list[str]) -> float:
    return sum(math.dist(cities[route[i]], cities[route[(i + 1) % len(route)]])
               for i in range(len(route)))

def two_opt_move(route: list[str], i: int, k: int) -> list[str]:
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]

def two_opt_move_random(route: list[str]) -> list[str]:
    i, k = random.sample(range(len(route)), 2)
    if i > k:
        i, k = k, i
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]

def tsp_brute_force():
    nodes = list(cities)
    n = len(nodes)

    if n <= 1:
        return nodes, 0.0

    start_node = nodes[0]
    remaining_nodes = nodes[1:]

    best_tour = None
    min_distance = float('inf')

    for perm in itertools.permutations(remaining_nodes):
        current_tour = [start_node] + list(perm)
        current_distance = tour_length(current_tour)

        if current_distance < min_distance:
            min_distance = current_distance
            best_tour = current_tour

    return best_tour

def linear_schedule(alpha: float):
    def linear(x: int | float):
        return max(0, x - alpha)
    return linear

def exponential_schedule(alpha: float):
    def exponential(x: int | float):
        return max(0, x * alpha)
    return exponential

def roll(n: float) -> bool:
    if n == 0:
        return False
    return random.random() < 1 / n

def simulated_annealing(route: list[str], schedule):
    current = list(route)
    t = 0
    T = 500
    graph = []
    graph.append((tour_length(current), t))
    while True:
        t += 1
        T = schedule(T)
        if t > 1000 or T <= 0:
            return current, graph
        new = two_opt_move_random(current)
        delta_E = tour_length(new) - tour_length(current)
        if delta_E <= 0:
            current = list(new)
        else:
            if roll(math.exp(-1 * delta_E / T)):
                current = list(new)
        graph.append((tour_length(current), t))

def random_route(seed: int):
    route = list(cities)
    random.shuffle(route)
    return route

route = random_route(0)

route = tsp_brute_force()
optimal_length = tour_length(route)
print(f"Optimal (brute): {route}  L = {optimal_length:.2f}")
print("")
t0 = time.perf_counter()
new_route, graph = simulated_annealing(route, linear_schedule(0.5))
t1 = time.perf_counter() - t0
print("Linear schedule 1:")
print(f"Before: {route}  L = {tour_length(route):.2f}")
print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")
print(f"Time:   {t1:.3f} s")
for pair in graph:
    if log_results: log1.write(f"{pair[0]},{pair[1]}\n")
if log_results: log5.write(f"{t1},{abs((optimal_length - tour_length(new_route)) / optimal_length) * 100}\n")
print("")
t0 = time.perf_counter()
new_route, graph = simulated_annealing(route, linear_schedule(0.9))
t1 = time.perf_counter() - t0
print("Linear schedule 2:")
print(f"Before: {route}  L = {tour_length(route):.2f}")
print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")
print(f"Time:   {t1:.3f} s")
for pair in graph:
    if log_results: log2.write(f"{pair[0]},{pair[1]}\n")
if log_results: log6.write(f"{t1},{abs((optimal_length - tour_length(new_route)) / optimal_length) * 100}\n")
print("")
t0 = time.perf_counter()
new_route, graph = simulated_annealing(route, exponential_schedule(0.5))
t1 = time.perf_counter() - t0
print("Exponential schedule 1:")
print(f"Before: {route}  L = {tour_length(route):.2f}")
print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")
print(f"Time:   {t1:.3f} s")
for pair in graph:
    if log_results: log3.write(f"{pair[0]},{pair[1]}\n")
if log_results: log7.write(f"{t1},{abs((optimal_length - tour_length(new_route)) / optimal_length) * 100}\n")
print("")
t0 = time.perf_counter()
new_route, graph = simulated_annealing(route, exponential_schedule(0.9))
t1 = time.perf_counter() - t0
print("Exponential schedule 2:")
print(f"Before: {route}  L = {tour_length(route):.2f}")
print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")
print(f"Time:   {t1:.3f} s")
for pair in graph:
    if log_results: log4.write(f"{pair[0]},{pair[1]}\n")
if log_results: log8.write(f"{t1},{abs((optimal_length - tour_length(new_route)) / optimal_length) * 100}\n")
print("")

if log_results: log1.close()
if log_results: log2.close()
if log_results: log3.close()
if log_results: log4.close()
if log_results: log5.close()
if log_results: log6.close()
if log_results: log7.close()
if log_results: log8.close()
