import random
import numpy as np



C1_min = C2_min = 1e-12 
C1_max = C2_max= 100e-12  
L_min = 1e-9  
L_max = 100e-9  

ZL = 50 + 60j

def fitness_function(x):
    C1 = x[0]
    C2 = x[1]
    L = x[2]
    Zin = calculate_impedance(C1,C2, L,f)
    VSWR = calculate_vswr(Zin)
    fitness = 1 / (VSWR + 1e-6)  
    return fitness

def calculate_impedance(C1,C2, L,f):
    #f=random.randrange(900e6, 1800e6, 25e6)
    s= 2* np.pi* f
    Zc1 = 1j / (s * C1)  
    Zc2 = 1j / (s* C2)  
    Zl = 1j * s * L  

    Rl = 50
    Xl = 60

    Ra = Rl/((s*C2*Rl)**2 + (1- (s*C2*Xl))**2)
    Xa = (Xl - s*C2*(Rl**2 + Xl**2))/((s*C2*Rl)**2 + (1 - s*C2*Xl)**2)


    Rb = Ra
    Xb = Xa + s*L

    Ri = Rb/((s*C1*Rb)**2 + (1-s*C1*Xb)**2)
    Xi = (Xb- s*C1*(Rb**2 + Xb**2))/((s*C1*Rb)**2 + (1-s*C1*Xb)**2)
    Zin= complex(Ri,Xi)
    return Zin


def calculate_vswr(Zin):
    coeff = (Zin- 50)/(Zin + 50)
    VSWR = (1 + abs(coeff)) / (1 - abs(coeff))
    return VSWR

def generate_population(size):
    population = []
    for i in range(size):
        C1 = random.uniform(C1_min, C1_max)
        C2 = random.uniform(C2_min, C2_max)
        L = random.uniform(L_min, L_max)
        individual = [C1,C2,L]
        population.append(individual)
    return population

def tournament_selection(population, k):
    parents = []
    for i in range(2):
        selected = random.sample(population, k)
        best = max(selected, key=lambda x: fitness_function(x))
        parents.append(best)
    return parents

def crossover(parents):
    offspring = []
    for i in range(len(parents[0])):
        if random.random() < 0.5:
            offspring.append(parents[0][i])
        else:
            offspring.append(parents[1][i])
    return offspring


def mutation(offspring):
    for i in range(len(offspring)):
        if random.random() < 0.1:  # mutation probability
            if i == 0:  # mutate capacitance
                offspring[i] = random.uniform(C1_min, C1_max)
            elif i==1:
                offspring[i] = random.uniform(C2_min, C2_max)
            else:  # mutate inductance
                offspring[i] = random.uniform(L_min, L_max)
    return offspring

def genetic_algorithm(population_size, k, num_generations):
    population = generate_population(population_size)
    for i in range(num_generations):
        parents = tournament_selection(population, k)
        offspring = crossover(parents)
        offspring = mutation(offspring)
        population.append(offspring)
        population = sorted(population, key=lambda x: fitness_function(x), reverse=True)
        population = population[:population_size]
        best_fitness = fitness_function(population[0])
    return population

import pandas as pd
import numpy as np

df= pd.DataFrame(columns=['Frequency',"C1","C2","L","Fitness_Value"])

i=0

while i<500:
  f=random.randrange(900e6, 1800e6, 25e6)
  population_size = 800
  k = 160
  num_generations = 1000
  population = genetic_algorithm(population_size, k, num_generations)


  best_solution = population[0]
  best_fitness = fitness_function(best_solution)


  print("Best solution: C1 = {}, C2 = {}, L = {}".format(best_solution[0], best_solution[1],best_solution[2]))
  print("Best fitness value: {}".format(best_fitness))
  if(best_fitness> 0.8):
    results=[]
    results.append([f,best_solution[0], best_solution[1], best_solution[2], best_fitness])
    print(len(results))
    df.loc[len(df)]= results[0]
  i=i+1



