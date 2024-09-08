import random
import numpy as np

# Parameters
NUM_CITIES = 20
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
GENERATIONS = 500

# Generate random coordinates for cities
cities = np.random.rand(NUM_CITIES, 2)

# Function to compute the total distance of a tour
def compute_total_distance(tour):
    distance = 0
    for i in range(len(tour)):
        start_city = cities[tour[i]]
        end_city = cities[tour[(i + 1) % len(tour)]]
        distance += np.linalg.norm(start_city - end_city)
    return distance

# Fitness function (inverse of the total distance)
def evaluate_fitness(tour):
    return 1 / compute_total_distance(tour)

# Initialise population with random tours
def initialise_population():
    return [list(np.random.permutation(NUM_CITIES)) for _ in range(POPULATION_SIZE)]

# Roulette wheel selection
def roulette_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probabilities = [score / total_fitness for score in fitness_scores]
    chosen_index = np.random.choice(len(population), p=selection_probabilities)
    return population[chosen_index]

# Ordered crossover
def perform_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(NUM_CITIES), 2))
    child = [None] * NUM_CITIES
    child[start:end] = parent1[start:end]
    current_position = 0
    for city in parent2:
        if city not in child:
            while child[current_position] is not None:
                current_position += 1
            child[current_position] = city
    return child

# Swap mutation
def apply_mutation(tour):
    for _ in range(NUM_CITIES):
        if random.random() < MUTATION_RATE:
            idx1, idx2 = random.sample(range(NUM_CITIES), 2)
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
    return tour

# Main genetic algorithm function
def run_genetic_algorithm():
    population = initialise_population()
    optimal_tour = None
    minimal_distance = float('inf')

    for generation in range(GENERATIONS):
        fitness_scores = [evaluate_fitness(tour) for tour in population]
        new_population = []

        for _ in range(POPULATION_SIZE // 2):
            parent1 = roulette_selection(population, fitness_scores)
            parent2 = roulette_selection(population, fitness_scores)
            offspring1 = apply_mutation(perform_crossover(parent1, parent2))
            offspring2 = apply_mutation(perform_crossover(parent1, parent2))
            new_population.extend([offspring1, offspring2])

        population = new_population

        current_best_tour = min(population, key=compute_total_distance)
        current_best_distance = compute_total_distance(current_best_tour)

        if current_best_distance < minimal_distance:
            minimal_distance = current_best_distance
            optimal_tour = current_best_tour

        print(f"Generation {generation + 1}: Shortest Distance = {minimal_distance}")

    return optimal_tour, minimal_distance

# Execute the genetic algorithm
if __name__ == "__main__":
    best_tour, best_distance = run_genetic_algorithm()
    print(f"Optimal Tour: {best_tour}\nShortest Distance: {best_distance}")
