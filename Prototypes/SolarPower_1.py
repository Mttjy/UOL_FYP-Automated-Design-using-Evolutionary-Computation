import random
import numpy as np

# Constants
POP_SIZE = 100
MUT_RATE = 0.01
GENS = 500
SOLAR_EFF = 0.2
PANEL_SIZE = 17.5
AVG_NEED = 15
LAND_SIZE = 21780

MAX_PANELS = int(LAND_SIZE / PANEL_SIZE)

# Fitness function: energy efficiency, environmental impact, space efficiency
def fitness(panel_count):
    energy_score = min(panel_count / AVG_NEED, 1)
    space_score = 1 - (panel_count * PANEL_SIZE / LAND_SIZE)
    env_score = 1 / panel_count
    return 0.4 * energy_score + 0.4 * space_score + 0.2 * env_score

# Create initial population
def create_population(size):
    return [random.randint(1, MAX_PANELS) for _ in range(size)]

# Selection function (roulette wheel)
def select(pop, fit):
    probs = [f / sum(fit) for f in fit]
    return np.random.choice(pop, p=probs)

# Crossover function (average crossover)
def crossover(p1, p2):
    return (p1 + p2) // 2

# Mutation function
def mutate(count):
    return max(1, count + random.randint(-1, 1) if random.random() < MUT_RATE else count)

# Main genetic algorithm
def genetic_algorithm():
    population = create_population(POP_SIZE)
    best_solution = max(population, key=fitness)

    for gen in range(GENS):
        fitnesses = [fitness(ind) for ind in population]
        new_population = [mutate(crossover(select(population, fitnesses), select(population, fitnesses))) for _ in range(POP_SIZE)]
        population = new_population
        current_best = max(population, key=fitness)
        if fitness(current_best) > fitness(best_solution):
            best_solution = current_best
        print(f"Gen {gen + 1}: Best Panels = {best_solution} (Fitness: {fitness(best_solution):.4f})")

    return best_solution

if __name__ == "__main__":
    optimal_panels = genetic_algorithm()
    print(f"Optimal Number of Panels: {optimal_panels}")
