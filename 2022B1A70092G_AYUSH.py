from CNF_Creator import *
import numpy as np
import time
import math

def createRandomPopulation (population_size = 20, num_symbols=50) -> np.ndarray:
    """
    This function generates a population of models, where each model has random values
    The population is stored as a 2D numpy array, where each row is a model

    Args:
        population_size : The size of the population to be generated
        num_symbols     : The number of different literals in the problem sentence

    Returns:
        population      : The randomly generated population
    """
    population = np.random.randint(0, 2, size=(population_size, num_symbols), dtype=np.int8)

    return population

def calculateFitness (individual, sentence) :
    """
    This function computes the fitness of an individual in a population for a given sentence
    The fitness function is simply the percentage of true clauses

    Args:
        individual  : The particular individual whose fitness must be calculated
        sentence    : The problem statement

    Returns:
        fitness     : The fitness of the individual for the problem statement
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

    Args:
        population  : The population to be evaluated
        sentence    : The problem statement

    Returns:
        fitness     : A 1D np.array containing the fitness values of each individual
    """
    return np.array([calculateFitness(ind, sentence) for ind in population])

def selectParentsRoulette (population, probabilities, num_parents=2) :
    """
    This function selects parents from a population with probabilities proportional to their fitness values
    and returns the parents (not their indices)

    Args:
        population      : The population from which the parents are to be selected
        probabilities   : A 1D array representing the normalised fitness for each individual
        num_parents     : The number of parents from which a single child is generated

    Returns:
        parents         : An array containing the individuals who are to be used as parents
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

def mutate (population, gen, stagnation_counter) :
    """
    This function performs random point mutations on the individuals of a population

    Args:
        population  : A 2D np array representing all the individuals of the population
        gen         : The current generation
        stagnation_counter : The number of generations for which the fitness value has not changed

    Returns:
        mutated_population : The mutated population 
    """
    sim_anneal_speed = 0.001
    mutation_rate = max(0.01, math.exp(-gen * sim_anneal_speed) * 0.5)

    if stagnation_counter > 50 :
        mutation_rate = min(0.8, mutation_rate+(stagnation_counter*0.01)) # temporary boost if stagnant
        # print(stagnation_counter, "------ Mutation boosted ------ ", gen)

    mutation_mask = np.random.rand(*population.shape) < mutation_rate
    mutated_population = population ^ mutation_mask

    return mutated_population

def generateNextPopulationEliteCulling (population, fitness, sentence, gen, stagnation_counter, elite_frac=0.1, cull_fact=2) :
    """
    This function generates the next population based on the current population and its fitness values. 
    It generates extra children and then discards the lowest performing ones.
    It also retains the highest performing individuals from the parent population

    Args:
        population  : A 2D np array containing all the individuals
        fitness     : A 1D array containing the fitness values of the corresponding individual
        sentence    : The problem statement expression
        gen         : The current generation
        stagnation_counter : The number of generations for which the fitness value has not changed
        elite_frac  : The fraction of best performing individuals who are retained into the next generation
        cull_fact   : The factor by which the population size is multiplied, to generate the next population, from which the worst performers are culled

    Returns:
        next_population : The 2D np array containing the individuals of the next generation
    """
    pop_size = len(population)
    elite_size = max(1, int(elite_frac * pop_size))
    to_cull_pop_size = int(cull_fact * pop_size)
    next_population = []

    elite_indices = np.argsort(fitness)[-elite_size:]
    elites = population[elite_indices]

    probabilities:np.ndarray
    if fitness.sum() == 0:
        probabilities = np.ones_like(fitness, dtype=float) / len(fitness)
    else :
        probabilities = fitness / fitness.sum()

    next_population = list(elites)

    while len(next_population) < (to_cull_pop_size - elite_size):
        parents = selectParentsRoulette(population, probabilities)
        children = crossover(parents, len(parents)) # generate same number of children as parents
        mutated_children = mutate(children, gen, stagnation_counter)
        next_population.extend(mutated_children)

    next_population = np.asarray(next_population[:to_cull_pop_size], dtype=population.dtype)

    # Evaluate children’s fitness
    child_fitness = np.array([calculateFitness(ind, sentence) for ind in next_population])

    # Select best pop_size from children
    best_indices = np.argsort(child_fitness)[-pop_size:]
    next_population = next_population[best_indices]

    return next_population

def main() :
    TIME_LIMIT = 44.8
    start_time = time.time()

    cnfC = CNF_Creator(n=50) # n is number of symbols in the 3-CNF sentence
    sentence = cnfC.CreateRandomSentence(m=120) # m is number of clauses in the 3-CNF sentence
    # print('Random sentence : ',sentence)

    # sentence = cnfC.ReadCNFfromCSVfile()
    # print('\nSentence from CSV file : ',sentence)

    max_generations = 99999999
    generation = 1
    best_fitness_history = [0.0]
    stagnation_counter = 0

    # Create population of models
    population_size = 300
    population = createRandomPopulation(population_size)

    # REPEAT till time = 45s or fitness values don't change or maximum number of generaions is reached
    while generation < max_generations and (time.time()-start_time < TIME_LIMIT):
        # Evaluate fitness of each model in the population
        fitness = evaluatePopulation(population, sentence)

        best_parent = np.argmax(fitness)
        best_fitness = fitness[best_parent]

        if abs(best_fitness-best_fitness_history[-1]) < 0.001:
            stagnation_counter += 1
        else :
            stagnation_counter = 0

        best_fitness_history.append(best_fitness)
        if generation%50 == 0:
            print("Generation ", generation)
            print("Best fitness value = ", best_fitness)
            print("Time : ", time.time()-start_time)
            print()

        # Check if SAT expression is solved
        if best_fitness == 1 :
            print("Solved in ", generation , " generations\n", population[best_parent])
            break

        # If not solved yet, generate the next population
        new_population = generateNextPopulationEliteCulling(population, fitness, sentence, generation, stagnation_counter, elite_frac=0.2, cull_fact=2)
        population = new_population

        generation += 1

    # print("Final population\n", population[0])




    # print('\n\n')
    # print('Roll No : 2020H1030999G')
    # print('Number of clauses in CSV file : ',len(sentence))
    # print('Best model : ',[1, -2, 3, -4, -5, -6, 7, 8, 9, 10, 11, 12, -13, -14, -15, -16, -17, 18, 19, -20, 21, -22, 23, -24, 25, 26, -27, -28, 29, -30, -31, 32, 33, 34, -35, 36, -37, 38, -39, -40, 41, 42, 43, -44, -45, -46, -47, -48, -49, -50])
    # print('Fitness value of best model : 99%')
    # print('Time taken : 5.23 seconds')
    # print('\n\n')
    
if __name__=='__main__':
    main()