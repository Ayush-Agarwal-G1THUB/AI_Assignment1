from CNF_Creator import *
import numpy as np
import time

def createRandomPopulation (population_size = 20, num_symbols=50) -> np.ndarray:
    """
    This function generates a population of models, where each model has random values
    The population is stored as a 2D numpy array, where each row is a model
    """
    population = np.random.randint(0, 2, size=(population_size, num_symbols), dtype=np.int8)

    return population

def fitnessFunction (individual, sentence) :
    """
    This function computes the fitness of an individual in a population for a given sentence
    The fitness function is simply the percentage of true clauses
    """
    fitness = 0
    for clause in sentence :
        for literal in clause :
            idx = abs(literal)
            if (literal>0 and individual[idx-1]==1) or (literal<0 and individual[idx-1]==0):
                fitness += 1
                break

    return fitness / len(sentence)

def evaluatePopulation (population, sentence) :
    """
    This function evaluates the fitness values for the entire population and returns it as a numpy array
    """
    return np.array([fitnessFunction(ind, sentence) for ind in population])

def selectParentsStochastic (population, probabilities, num_parents=2) :
    """
    This function selects parents from a population with probabilities proportional to their fitness values
    and returns the parents (not their indices)
    """
    parents_idx = np.random.choice(len(population), size=num_parents, p=probabilities)

    return population[parents_idx]

def crossover(parents, num_offspring=1, crossover_type="single", num_points=1):
    """
    Generalized crossover operator.
    
    Args:
        parents (np.ndarray): 2D array of shape (num_parents, chromosome_length).
        num_offspring (int): Number of offspring to produce.
        crossover_type (str): "single" or "multi".
        num_points (int): Number of crossover points (used if crossover_type == "multi").
    
    Returns:
        np.ndarray: 2D array of shape (num_offspring, chromosome_length).
    """

    num_parents, chromosome_length = parents.shape
    offspring = np.empty((num_offspring, chromosome_length), dtype=parents.dtype)

    for i in range(num_offspring):
        # Select which parents will contribute to this offspring
        chosen_parents_idx = np.random.choice(num_parents, size=num_points+1, replace=True)
        cut_points = []

        if crossover_type == "single":
            cut_points = [np.random.randint(1, chromosome_length)]
        elif crossover_type == "multi":
            cut_points = np.sort(np.random.choice(range(1, chromosome_length), size=num_points, replace=False))

        cut_points = [0] + cut_points + [chromosome_length]

        # Alternate between chosen parents across segments
        child = []
        for j in range(len(cut_points)-1):
            parent_idx = chosen_parents_idx[j % len(chosen_parents_idx)]
            start, end = cut_points[j], cut_points[j+1]
            child.extend(parents[parent_idx, start:end])

        offspring[i, :] = child

    return offspring

def mutate (children, mutation_rate=0.01) :
    mutation_mask = np.random.rand(*children.shape) < mutation_rate
    mutated_children = children ^ mutation_mask

    return mutated_children

def generateNextPopulationCulling (population, fitness, sentence, cull_frac=2):
    """
    This function generates the next population based on the current population and its fitness values
    It generates extra children and then discards the lowest performing ones
    """
    pop_size = len(population)
    generated_pop_size = int(cull_frac * pop_size)
    next_population = []

    probabilities:np.ndarray
    if fitness.sum() == 0:
        probabilities = np.ones_like(fitness, dtype=float) / len(fitness)
    else :
        probabilities = fitness / fitness.sum()

    while len(next_population) < generated_pop_size:
        parents = selectParentsStochastic(population, probabilities)
        children = crossover(parents, len(parents)) # generate same number of children as parents
        mutated_children = mutate(children)
        next_population.extend(mutated_children)

    next_population = np.asarray(next_population[:generated_pop_size], dtype=population.dtype)

    # Evaluate children’s fitness
    child_fitness = np.array([fitnessFunction(ind, sentence) for ind in next_population])

    # Select best pop_size from children
    best_indices = np.argsort(child_fitness)[-pop_size:]
    next_population = next_population[best_indices]

    return next_population

def generateNextPopulationElitism (population, fitness, elite_frac=0.2) :
    """
    This function generates the next population based on the current population and its fitness values
    It also employs elitism, retaining the top elite_frac percentage of individuals in the parent population
    """
    pop_size = len(population)
    elite_size = max(1, int(elite_frac * pop_size)) # Ensure at least 1 elite

    # Find individuals with highest fitness values
    elite_indices = np.argsort(fitness)[-elite_size:]
    elites = population[elite_indices]

    probabilities:np.ndarray
    if fitness.sum() == 0:
        probabilities = np.ones_like(fitness, dtype=float) / len(fitness)
    else :
        probabilities = fitness / fitness.sum()

    # Add the elites into the new population
    next_population = list(elites)

    # Add children into the population
    while len(next_population) < pop_size:
        parents = selectParentsStochastic(population, probabilities)
        children = crossover(parents, len(parents)) # generate same number of children as parents
        mutated_children = mutate(children)
        next_population.extend(mutated_children)

    return np.asarray(next_population[:pop_size], dtype=population.dtype)

def generateNextPopulationBasic (population, fitness) :
    """
    This function generates the next population based on the current population and its fitness values
    """
    pop_size = len(population)
    next_population = []

    probabilities:np.ndarray
    if fitness.sum() == 0:
        probabilities = np.ones_like(fitness, dtype=float) / len(fitness)
    else :
        probabilities = fitness / fitness.sum()

    while len(next_population) < pop_size:
        parents = selectParentsStochastic(population, probabilities)
        children = crossover(parents, len(parents)) # generate same number of children as parents
        mutated_children = mutate(children)
        next_population.extend(mutated_children)

    return np.asarray(next_population[:pop_size], dtype=population.dtype)

def main() :
    start_time = time.time()
    curr_time = time.time()

    cnfC = CNF_Creator(n=50) # n is number of symbols in the 3-CNF sentence
    sentence = cnfC.CreateRandomSentence(m=200) # m is number of clauses in the 3-CNF sentence
    print('Random sentence : ',sentence)

    max_generations = 99999999
    gen = 1

    # Create population of models
    population_size = 200
    population = createRandomPopulation(population_size)

    # REPEAT till time = 45s or fitness values don't change or maximum number of generaions is reached
    while gen < max_generations and (curr_time-start_time < 44.8):
        # Evaluate fitness of each model in the population
        fitness = evaluatePopulation(population, sentence)

        best_parent = np.argmax(fitness)
        best_fitness = fitness[best_parent]
        if gen%5 == 0:
            print("Best fitness value = ", best_fitness)
            print("Time : ", curr_time-start_time)

        # Check if SAT expression is solved
        if best_fitness == 1 :
            print("Solved!\n", population[best_parent])
            break

        # If not solved yet, generate the next population
        new_population = generateNextPopulationCulling(population, fitness, sentence)
        population = new_population

        gen += 1
        curr_time = time.time()

    # print("Final population\n", population[0])



    # sentence = cnfC.ReadCNFfromCSVfile()
    # print('\nSentence from CSV file : ',sentence)

    # print('\n\n')
    # print('Roll No : 2020H1030999G')
    # print('Number of clauses in CSV file : ',len(sentence))
    # print('Best model : ',[1, -2, 3, -4, -5, -6, 7, 8, 9, 10, 11, 12, -13, -14, -15, -16, -17, 18, 19, -20, 21, -22, 23, -24, 25, 26, -27, -28, 29, -30, -31, 32, 33, 34, -35, 36, -37, 38, -39, -40, 41, 42, 43, -44, -45, -46, -47, -48, -49, -50])
    # print('Fitness value of best model : 99%')
    # print('Time taken : 5.23 seconds')
    # print('\n\n')
    
if __name__=='__main__':
    main()