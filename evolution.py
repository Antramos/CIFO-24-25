import random
import numpy as np
from operators import (
    mutate_swap,
    mutate_team_shift,
    mutate_shuffle_team,
    crossover_one_point,
    crossover_uniform,
    selection_tournament,
    selection_ranking
)
from solution import LeagueSolution, LeagueHillClimbingSolution

# Map operator names to functions for flexible configuration
MUTATIONS = [mutate_swap, mutate_team_shift, mutate_shuffle_team]
CROSSOVERS = [crossover_one_point, crossover_uniform]
SELECTIONS = [selection_tournament, selection_ranking]

def generate_population(players, size):
    population = []
    while len(population) < size:
        candidate = LeagueSolution()
        if candidate.is_valid(players):
            population.append(candidate)
    return population

def genetic_algorithm(
    players,
    population_size=50,
    generations=30,
    mutation_rate=0.2,
    elite_size=5,
    mutation_operator=mutate_swap,
    crossover_operator=crossover_one_point,
    selection_operator=selection_tournament
):
    population = generate_population(players, population_size)
    history = []
    best_solution = min(population, key=lambda s: s.fitness(players))

    for gen in range(generations):
        new_population = []
        population.sort(key=lambda x: x.fitness(players))
        new_population.extend(population[:elite_size])

        while len(new_population) < population_size:
            # No loop principal do GA
            parent1 = selection_operator(population, players)
            parent2 = selection_operator(population, players)
            
            #parent1 = selection_operator(population)
            #parent2 = selection_operator(population)
            child = crossover_operator(parent1, parent2)

            if random.random() < mutation_rate:
                child = mutation_operator(child)

            if child.is_valid(players):
                new_population.append(child)

        population = new_population
        current_best = min(population, key=lambda s: s.fitness(players))
        if current_best.fitness(players) < best_solution.fitness(players):
            best_solution = current_best
        history.append(best_solution.fitness(players))

    return best_solution, history


def hill_climbing(players, max_iterations=1000, verbose=False):
    current = LeagueHillClimbingSolution()
    while not current.is_valid(players):
        current = LeagueHillClimbingSolution()

    current_fitness = current.fitness(players)
    history = [current_fitness]

    for iteration in range(max_iterations):
        neighbors = current.get_neighbors(players)
        if not neighbors:
            break

        neighbor = min(neighbors, key=lambda x: x.fitness(players))
        neighbor_fitness = neighbor.fitness(players)

        if neighbor_fitness < current_fitness:
            current = neighbor
            current_fitness = neighbor_fitness
            history.append(current_fitness)
            if verbose:
                print(f"Iteration {iteration}: fitness = {current_fitness}")
        else:
            break

    return current, current_fitness, history

import numpy as np 

def simulated_annealing(players, max_iterations=1000, initial_temp=100.0, cooling_rate=0.995, verbose=False):
    num_players = len(players)

    # Retry until valid initial solution
    while True:
        current_solution = LeagueSolution([random.randint(0, 4) for _ in range(num_players)])
        if current_solution.is_valid(players):
            break

    current_fitness = current_solution.fitness(players)
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    history = [current_fitness]

    temp = initial_temp
    iteration = 0

    while iteration < max_iterations and temp > 0.1:
        neighbor = mutate_swap(current_solution)
        if not neighbor.is_valid(players):
            continue

        neighbor_fitness = neighbor.fitness(players)
        delta = neighbor_fitness - current_fitness

        if delta <= 0 or random.random() < np.exp(-delta / temp):
            current_solution = neighbor
            current_fitness = neighbor_fitness
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness

        history.append(current_fitness)
        temp *= cooling_rate
        iteration += 1

        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: fitness = {current_fitness:.6f}, temp = {temp:.2f}")

    return best_solution, history, best_fitness








