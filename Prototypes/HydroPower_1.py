import random
import numpy as np

# Constants
POP_SIZE = 100
MUT_RATE = 0.05
GENS = 500
WATER_FLOW_RANGE = (0.5, 2.0)  # Range of water flow rates in cubic meters per second (m³/s)
EFFICIENCY_RANGE = (0.6, 0.9)  # Range of turbine efficiencies
AVG_NEED = 1.21375  # Average power needs per hour in kilowatts (kW)
LAND_SIZE = 21780  # Size of land available in square meters (m²)

# Calculate maximum number of turbines based on available land
MAX_TURBINES = int(LAND_SIZE / 10)  # Assuming each turbine requires 10 m²

# Create initial population with random water flow rates and turbine counts
def create_population(size):
    return [(random.uniform(*WATER_FLOW_RANGE), random.randint(1, MAX_TURBINES)) for _ in range(size)]

# Fitness function with variable water flow and turbine count
def fitness(water_flow, turbine_count):
    power_output = water_flow * turbine_count * random.uniform(*EFFICIENCY_RANGE) * 9.81 * 3600 * 24 / 1000  # Power output in kilowatt-hours per day (kWh/day)
    energy_score = min(power_output / AVG_NEED, 1)
    space_score = 1 - (turbine_count * 10 / LAND_SIZE)  # Assuming each turbine requires 10 m²
    env_score = 1 / turbine_count
    return 0.4 * energy_score + 0.4 * space_score + 0.2 * env_score

# Selection function (roulette wheel)
def select(pop, fit):
    probs = [f / sum(fit) for f in fit]
    return pop[np.random.choice(len(pop), p=probs)]

# Crossover function (average crossover)
def crossover(p1, p2):
    return (p1[0], p2[1])  # Swap characteristics between parents

# Mutation function
def mutate(turbine_count):
    if random.random() < MUT_RATE:
        turbine_count += random.randint(-3, 3)  # Larger range for mutation
    return max(1, min(turbine_count, MAX_TURBINES))

# Main genetic algorithm
def genetic_algorithm():
    population = create_population(POP_SIZE)
    best_solution = max(population, key=lambda x: fitness(x[0], x[1]))
    best_fitness = fitness(best_solution[0], best_solution[1])
    no_improvement_count = 0
    max_no_improvement = 10

    for gen in range(GENS):
        if no_improvement_count >= max_no_improvement:
            print("Stopping early due to lack of improvement")
            break
        
        fitnesses = [fitness(ind[0], ind[1]) for ind in population]
        new_population = []

        for _ in range(POP_SIZE // 2):
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            offspring1 = (parent1[0], mutate(crossover(parent1, parent2)[1]))
            offspring2 = (parent2[0], mutate(crossover(parent1, parent2)[1]))
            new_population.extend([offspring1, offspring2])

        population = new_population
        current_best = max(population, key=lambda x: fitness(x[0], x[1]))
        current_fitness = fitness(current_best[0], current_best[1])

        if current_fitness > best_fitness:
            best_solution = current_best
            best_fitness = current_fitness
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        print(f"Gen {gen + 1}: Best Solution = {best_solution} (Fitness: {best_fitness:.4f})")
        if best_fitness == 1.0:
            print(f"Optimal solution found in generation {gen + 1}")
            break

    return best_solution[1]  # Return only the optimal number of turbines

if __name__ == "__main__":
    optimal_turbines = genetic_algorithm()
    print(f"Optimal Number of Turbines: {optimal_turbines}")
