import pandas as pd
import numpy as np
from deap import base, creator, tools

# Load the dataset
data = pd.read_csv("cleaned_snwdata.csv")

# Convert the 'Date and Hour' column to datetime format with UTC
data['Date and Hour'] = pd.to_datetime(data['Date and Hour'], utc=True)

# Group by date and energy source, then sum the production
daily_totals = data.groupby([data['Date and Hour'].dt.date, 'Source'])['Production'].sum().reset_index()

# Pivot the table to get daily totals for each energy source
daily_totals_pivot = daily_totals.pivot(index='Date and Hour', columns='Source', values='Production').fillna(0)

# Calculate average production rates
SOLAR_PANEL_PRODUCTION_RATE = daily_totals_pivot['Solar'].mean()
WIND_TURBINE_PRODUCTION_RATE = daily_totals_pivot['Wind'].mean()

# Define the maximum number of solar panels and wind turbines that can be placed
max_solar_panels = 5000  
max_wind_turbines = 10  

# Define the total available area
TOTAL_AREA = 28560

# Define the width and length of solar panels and wind turbines
SOLAR_PANEL_WIDTH = 2  
SOLAR_PANEL_LENGTH = 4 
WIND_TURBINE_WIDTH = 40 
WIND_TURBINE_LENGTH = 80 

# Fitness evaluation function
def evaluate(individual):
    num_solar_panels, num_wind_turbines = individual
    
    # Ensure number of panels and turbines do not exceed max constraints
    if num_solar_panels > max_solar_panels or num_wind_turbines > max_wind_turbines:
        return (-1e10,)  # Large negative fitness to penalise exceeding max constraints
    
    area_solar_panels = num_solar_panels * (SOLAR_PANEL_WIDTH * SOLAR_PANEL_LENGTH)
    area_wind_turbines = num_wind_turbines * (WIND_TURBINE_WIDTH * WIND_TURBINE_LENGTH)
    
    total_area_used = area_solar_panels + area_wind_turbines
    
    # Check if total area exceeds available area
    if total_area_used > TOTAL_AREA:
        return (-1e10,)  # Large negative fitness to penalise exceeding the area
    
    # Calculate total energy production
    total_energy_production = (
        num_solar_panels * SOLAR_PANEL_PRODUCTION_RATE +
        num_wind_turbines * WIND_TURBINE_PRODUCTION_RATE
    )
    
    return (total_energy_production,)

# Mutation function ensuring constraints
def mutate(individual):
    if np.random.rand() < 0.5:
        # Mutate the number of solar panels
        individual[0] = np.random.randint(1, max_solar_panels + 1)
    else:
        # Mutate the number of wind turbines
        individual[1] = np.random.randint(1, max_wind_turbines + 1)
    return individual,

# Set up the genetic algorithm framework
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", np.random.randint, 1, 100)  # Random integer between 1 and 100
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the evolution process and store results in a DataFrame
def run_evolution(run_number):
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda values: sum(values) / len(values))
    stats.register("min", min)
    stats.register("max", max)
    
    max_generations = 1000
    
    # Evolution process
    for gen in range(max_generations):
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            if np.random.rand() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Replace the population with the offspring
        pop[:] = offspring
        
        # Re-evaluate fitness for invalid individuals
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)
        
        # Update the Hall of Fame
        hof.update(pop)
        
        # Collect stats for current generation
        record = stats.compile(pop)
        print(f"Run {run_number} - Generation {gen}: {record}")
    
    # Return the best individual from this run
    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]
    num_solar_panels, num_wind_turbines = best_individual
    
    return {
        'Run': run_number,
        'Num Solar Panels': num_solar_panels,
        'Num Wind Turbines': num_wind_turbines,
        'Best Fitness (Energy Production)': best_fitness
    }

# Process the top results and save them into a single CSV file
def process_best_results(all_run_results, top_n=5):
    # Convert results into a DataFrame
    combined_results = pd.DataFrame(all_run_results)
    
    # Sort by the 'Best Fitness (Energy Production)' in descending order
    sorted_results = combined_results.sort_values(by='Best Fitness (Energy Production)', ascending=False)
    
    # Drop duplicates based on the number of solar panels and wind turbines, keeping the best unique solutions
    unique_sorted_results = sorted_results.drop_duplicates(subset=['Num Solar Panels', 'Num Wind Turbines'])
    
    # Select the top N solutions
    top_solutions = unique_sorted_results.head(top_n)
    
    # Save the top solutions to a new CSV file
    top_solutions.to_csv("top_solutions.csv", index=False)

def main():
    all_run_results = []
    
    # Run the genetic algorithm 5 times
    for run_number in range(1, 6):
        best_result = run_evolution(run_number)
        all_run_results.append(best_result)
    
    process_best_results(all_run_results, top_n=5)

if __name__ == "__main__":
    main()
