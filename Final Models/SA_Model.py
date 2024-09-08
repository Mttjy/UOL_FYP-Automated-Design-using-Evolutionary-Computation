import numpy as np
import pandas as pd
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
MAX_SOLAR_PANEL_DISTANCE = 1 
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

    # Check if solar panels are within maximum distance and reward accordingly
    systematic_reward = 0
    for i in range(len(solar_panels) - 1):
        if distance(solar_panels[i], solar_panels[i + 1]) <= MAX_SOLAR_PANEL_DISTANCE:
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

    return total_fitness

def create_individual(num_solar_panels, num_wind_turbines):
    """Generate an individual with solar panels and wind turbines placed in an orderly fashion."""
    grid = create_grid()
    solar_panels = []
    wind_turbines = []

    pos = (0, 0)
    for _ in range(num_solar_panels):
        if place_item(grid, SOLAR_PANEL_SIZE, pos, 1):
            solar_panels.append(pos)
            # Move downwards for the next solar panel
            next_pos_y = pos[1] + SOLAR_PANEL_SIZE[0]
            # If the column is filled, move to the next column
            if next_pos_y + SOLAR_PANEL_SIZE[0] > GRID_HEIGHT:
                next_pos_y = 0
                pos = (pos[0] + SOLAR_PANEL_SIZE[1], next_pos_y)
                # If we've moved beyond the grid width, start wrapping around
                if pos[0] + SOLAR_PANEL_SIZE[1] > GRID_WIDTH:
                    pos = (0, 0)
            else:
                pos = (pos[0], next_pos_y)

    count = 0
    turbine_sizes = [WIND_TURBINE_SIZE_1, WIND_TURBINE_SIZE_2]
    size_index = 0 

    while count < num_wind_turbines and size_index < len(turbine_sizes):
        turbine_size = turbine_sizes[size_index]
        placed = False
        
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                pos = (col, row)
                if count < num_wind_turbines:
                    if place_item(grid, turbine_size, pos, 2):
                        wind_turbines.append(pos)
                        count += 1
                        placed = True
        
        # If no turbines were placed, try the next size
        if not placed:
            size_index += 1

    return solar_panels + wind_turbines

def simulated_annealing(num_solar_panels, num_wind_turbines, initial_temp=1000, cooling_rate=0.99, max_iterations=1000):
    """Run Simulated Annealing for placing solar panels and wind turbines."""
    current_solution = create_individual(num_solar_panels, num_wind_turbines)
    current_fitness = evaluate(current_solution, num_solar_panels, num_wind_turbines)
    best_solution = list(current_solution)
    best_fitness = current_fitness

    temp = initial_temp

    for iteration in range(max_iterations):
        new_solution = list(current_solution)
        # Randomly choose to add or remove an item
        if np.random.rand() < 0.5 and len(new_solution) > 0:
            # Remove an item
            idx = np.random.randint(len(new_solution))
            new_solution.pop(idx)
        else:
            # Add a new item
            if len(new_solution) < num_solar_panels + num_wind_turbines:
                x = np.random.randint(0, GRID_WIDTH)
                y = np.random.randint(0, GRID_HEIGHT)
                new_solution.append((x, y))
        
        new_fitness = evaluate(new_solution, num_solar_panels, num_wind_turbines)

        # Acceptance probability
        if new_fitness > current_fitness or np.random.rand() < np.exp((new_fitness - current_fitness) / temp):
            current_solution = new_solution
            current_fitness = new_fitness

            if current_fitness > best_fitness:
                best_solution = list(current_solution)
                best_fitness = current_fitness

        # Cool down
        temp *= cooling_rate

        print(f"Iteration {iteration}: Best Fitness = {best_fitness}, Current Fitness = {current_fitness}")

    # Plot the best solution
    grid = create_grid()
    solar_panels = best_solution[:num_solar_panels]
    wind_turbines = best_solution[num_solar_panels:num_solar_panels + num_wind_turbines]

    # Place solar panels
    for pos in solar_panels:
        place_item(grid, SOLAR_PANEL_SIZE, pos, 1)
    
    # Place wind turbines
    for pos in wind_turbines:
        if not place_item(grid, WIND_TURBINE_SIZE_1, pos, 2):
            place_item(grid, WIND_TURBINE_SIZE_2, pos, 2)

    plot_grid(grid, 'SA_best_placement.png')
    print("Best Placement Saved as 'SA_best_placement.png'")

def main():
    """Main function to read solutions and run simulated annealing."""
    # Read the CSV file
    top_solutions = pd.read_csv('top_solutions.csv')
    
    # Sort by fitness or another criterion to select the top entries
    top_solutions_sorted = top_solutions.sort_values(by='Best Fitness (Energy Production)', ascending=False)
    
    # Select the top N solutions
    top_n = 1
    top_solutions_to_run = top_solutions_sorted.head(top_n)
    
    # Run simulated annealing for each top solution
    for _, row in top_solutions_to_run.iterrows():
        run_number = int(row['Run'])
        num_solar_panels = int(row['Num Solar Panels'])
        num_wind_turbines = int(row['Num Wind Turbines'])
        
        print(f"Running simulation for Run {run_number} with {num_solar_panels} solar panels and {num_wind_turbines} wind turbines...")
        simulated_annealing(num_solar_panels, num_wind_turbines)

if __name__ == "__main__":
    main()
