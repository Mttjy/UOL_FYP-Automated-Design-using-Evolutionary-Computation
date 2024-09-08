import random
import string

# Define parameters
POPULATION_SIZE = 10
GENOME_LENGTH = 8
GENES = "01"  # Binary genes
MUTATION_RATE = 0.01
GENERATIONS = 50 #MAX Test Stopping Point
TOURNAMENT_SIZE = 3

TARGET_GENOME = "11111111"

# Dynamically generate a target genome if not defined
if not TARGET_GENOME:
    TARGET_GENOME = ''.join(random.choice(GENES) for _ in range(GENOME_LENGTH))

# Randomised genome
def create_genome():
    return ''.join(random.choice(GENES) for _ in range(GENOME_LENGTH))

# Initial population creation
def create_population(size):
    return [create_genome() for _ in range(size)]

# Fitness function: number of matching genes with the target genome
def fitness(genome):
    return sum(1 for a, b in zip(genome, TARGET_GENOME) if a == b)

# Selection function: tournament selection
def selection(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    return max(tournament, key=fitness)

# Crossover function: single-point crossover
def crossover(parent1, parent2):
    if parent1 == parent2:
        return parent1  # Prevents identical crossover
    point = random.randint(1, GENOME_LENGTH - 1)
    return parent1[:point] + parent2[point:]

# Mutation function: random bit flip
def mutate(genome):
    genome = list(genome)
    for i in range(GENOME_LENGTH):
        if random.random() < MUTATION_RATE:
            genome[i] = random.choice(GENES)
    return ''.join(genome)

# Main genetic algorithm
def genetic_algorithm():
    population = create_population(POPULATION_SIZE)
    for generation in range(GENERATIONS):
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1 = selection(population)
            parent2 = selection(population)
            offspring1 = mutate(crossover(parent1, parent2))
            offspring2 = mutate(crossover(parent1, parent2))
            new_population.extend([offspring1, offspring2])
        population = new_population
        best_genome = max(population, key=fitness)
        print(f"Generation {generation + 1}: {best_genome} (Fitness: {fitness(best_genome)})")
        if fitness(best_genome) == GENOME_LENGTH:
            print(f"Optimal solution found in generation {generation + 1}")
            break

# Run the genetic algorithm
if __name__ == "__main__":
    genetic_algorithm()
