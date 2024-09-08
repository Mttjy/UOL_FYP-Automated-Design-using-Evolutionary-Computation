import numpy as np
import random
import pandas as pd
from deap import base, creator, tools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define grid size
GRID_WIDTH = 220
GRID_HEIGHT = 140

# Define dimensions of solar panels and wind turbines in grid units
SOLAR_PANEL_SIZE = (2, 4)
WIND_TURBINE_SIZE_1 = (80, 40)  
WIND_TURBINE_SIZE_2 = (40, 80) 
MIN_WIND_TURBINE_DISTANCE = 5
PENALTY_INSUFFICIENT_WIND_TURBINES = -2000

def create_grid():
    """Create a new grid with all cells set to 0 (free)."""
    return np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

def place_item(grid, item_size, position, item_type):
    """Place an item on the grid if it fits."""
    item_height, item_width = item_size
    x, y = position
    if (x + item_width <= GRID_WIDTH) and (y + item_height <= GRID_HEIGHT):
        if np.all(grid[y:y + item_height, x:x + item_width] == 0):
            grid[y:y + item_height, x:x + item_width] = item_type
            return True
    return False

def remove_item(grid, item_size, position):
    """Remove an item from the grid."""
    item_height, item_width = item_size
    x, y = position
    grid[y:y + item_height, x:x + item_width] = 0

def plot_grid(grid, filename):
    """Visualise the grid and save as an image."""
    # Define a custom colormap
    cmap = mcolors.ListedColormap(['white', 'blue', 'lightgreen'])
    bounds = [0, 1, 2, 3]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=cmap, norm=norm, interpolation='none')
    plt.title("Best Placement of Solar Panels and Wind Turbines")
    plt.colorbar(label='Grid Value')
    plt.savefig(filename)
    plt.close()

def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def evaluate(individual, num_solar_panels, num_wind_turbines):
    grid = create_grid()
    total_fitness = 0
    solar_panels = individual[:num_solar_panels]
    wind_turbines = individual[num_solar_panels:num_solar_panels + num_wind_turbines]
    
    # Place solar panels
    for pos in solar_panels:
        if place_item(grid, SOLAR_PANEL_SIZE, pos, 1):
            total_fitness += 1  # Reward for placing a solar panel

    # Check if solar panels are adjacent and reward accordingly
    systematic_reward = 0
    for i in range(len(solar_panels) - 1):
        if (solar_panels[i][0] + SOLAR_PANEL_SIZE[1] == solar_panels[i + 1][0] and solar_panels[i][1] == solar_panels[i + 1][1]) or \
           (solar_panels[i][1] + SOLAR_PANEL_SIZE[0] == solar_panels[i + 1][1] and solar_panels[i][0] == solar_panels[i + 1][0]):
            systematic_reward += 1
    total_fitness += systematic_reward

    # Place wind turbines
    placed_wind_turbines = 0
    for pos in wind_turbines:
        if place_item(grid, WIND_TURBINE_SIZE_1, pos, 2) or place_item(grid, WIND_TURBINE_SIZE_2, pos, 2):
            total_fitness += 2  # Reward for placing a wind turbine
            placed_wind_turbines += 1
    
    # Penalise wind turbines that are too close to each other
    for i in range(len(wind_turbines)):
        for j in range(i + 1, len(wind_turbines)):
            if distance(wind_turbines[i], wind_turbines[j]) < MIN_WIND_TURBINE_DISTANCE:
                total_fitness -= 5  # Penalise for being too close
    
    # Penalise if total area exceeds available area or unused space
    total_area_used = np.sum(grid)
    if total_area_used > GRID_WIDTH * GRID_HEIGHT:
        total_fitness -= 1000  # Penalise large negative fitness
    
    empty_space = np.sum(grid == 0)
    total_fitness -= empty_space * 0.01

    # Apply penalty for insufficient wind turbines
    if placed_wind_turbines < num_wind_turbines:
        total_fitness += PENALTY_INSUFFICIENT_WIND_TURBINES

    return total_fitness,

def create_individual(num_solar_panels, num_wind_turbines):
    """Generate an individual with solar panels and wind turbines placed in an orderly fashion."""
    grid = create_grid()
    solar_panels = []
    wind_turbines = []

    # Place solar panels first
    num_cols_solar = GRID_HEIGHT // SOLAR_PANEL_SIZE[0]
    num_rows_solar = GRID_WIDTH // SOLAR_PANEL_SIZE[1]
    
    count = 0
    for col in range(num_rows_solar):
        for row in range(num_cols_solar):
            if count < num_solar_panels:
                pos = (col * SOLAR_PANEL_SIZE[1], row * SOLAR_PANEL_SIZE[0])
                if place_item(grid, SOLAR_PANEL_SIZE, pos, 1):
                    solar_panels.append(pos)
                    count += 1

    # Place wind turbines with both sizes
    count = 0
    turbine_sizes = [WIND_TURBINE_SIZE_1, WIND_TURBINE_SIZE_2]
    size_index = 0  # Start with the first size

    while count < num_wind_turbines and size_index < len(turbine_sizes):
        turbine_size = turbine_sizes[size_index]
        placed = False
        
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                if count < num_wind_turbines:
                    pos = (col, row)
                    if place_item(grid, turbine_size, pos, 2):
                        wind_turbines.append(pos)
                        count += 1
                        placed = True
        
        # If no turbines were placed, try alternate size
        if not placed:
            size_index += 1
    
    return solar_panels + wind_turbines

def run_evolution(num_solar_panels, num_wind_turbines):
    """Run the genetic algorithm for placing solar panels and wind turbines."""
    # Define the fitness function and individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual(num_solar_panels, num_wind_turbines))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: evaluate(ind, num_solar_panels, num_wind_turbines))
    toolbox.register("mate", tools.cxUniform, indpb=0.5)  # Uniform crossover for discrete values
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)  # Shuffle mutation for discrete values
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create population
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    
    # Set up statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else float('-inf'))
    stats.register("avg", lambda values: sum(values) / len(values))
    stats.register("min", min)
    stats.register("max", max)
    
    max_generations = 250
    stagnation_limit = 10  # Allow fitness to stagnate for 10 generations
    stagnation_count = 0  # Counter for stagnating generations
    last_max_fitness = None

    # Evolution process
    for gen in range(max_generations):
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if np.random.rand() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        pop[:] = offspring
        hof.update(pop)
        record = stats.compile(pop)
        print(f"Generation {gen}: {record}")

        # Check for stagnation
        current_max_fitness = record['max']
        if last_max_fitness is not None and current_max_fitness == last_max_fitness:
            stagnation_count += 1
        else:
            stagnation_count = 0  # Reset stagnation counter if fitness improves
        last_max_fitness = current_max_fitness
        
        # Trigger diversification if stagnation persists
        if stagnation_count >= stagnation_limit:
            print(f"Fitness stagnated for {stagnation_limit} generations. Increasing mutation rate...")
            toolbox.unregister("mutate")
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)  # Increase mutation probability
            stagnation_count = 0  # Reset stagnation counter

    # Get the best individual and plot the grid
    best_individual = hof[0]
    grid = create_grid()
    solar_panels = best_individual[:num_solar_panels]
    wind_turbines = best_individual[num_solar_panels:num_solar_panels + num_wind_turbines]
    
    # Place solar panels
    for pos in solar_panels:
        place_item(grid, SOLAR_PANEL_SIZE, pos, 1)
    
    # Place wind turbines
    for pos in wind_turbines:
        if not place_item(grid, WIND_TURBINE_SIZE_1, pos, 2):
            place_item(grid, WIND_TURBINE_SIZE_2, pos, 2)

    plot_grid(grid, 'GA_best_placement.png')
    print("Best Placement Saved as 'GA_best_placement.png'")


def main():
    """Main function to read solutions and run evolution."""
    # Read the CSV file
    top_solutions = pd.read_csv('top_solutions.csv')
    
    # Sort by fitness or another criterion to select the top entries
    top_solutions_sorted = top_solutions.sort_values(by='Best Fitness (Energy Production)', ascending=False)
    
    # Select the top N solutions
    top_n = 1  # Adjust this number as needed
    top_solutions_to_run = top_solutions_sorted.head(top_n)
    
    # Run evolution for each top solution
    for _, row in top_solutions_to_run.iterrows():
        run_number = int(row['Run'])
        num_solar_panels = int(row['Num Solar Panels'])
        num_wind_turbines = int(row['Num Wind Turbines'])
        
        print(f"Running simulation for Run {run_number} with {num_solar_panels} solar panels and {num_wind_turbines} wind turbines...")
        run_evolution(num_solar_panels, num_wind_turbines)

if __name__ == "__main__":
    main()
