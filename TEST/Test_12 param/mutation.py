import numpy as np
from complex_data import complex_data
import math
from global_configuration import global_configuration
import random
from Fitness import Fitness
from scipy.stats import norm

def Mutate(x, mu, VarMin, VarMax):
	nVar = len(x)
	nmu = math.ceil(mu * nVar)
	np.random.seed(1)
	j = random.randint(min(nVar, nmu), max(nVar, nmu))
	sigma = 0.1 * (VarMax - VarMin)
	y = x.copy()
	y = np.round(x + sigma * norm.ppf(np.random.rand(len(x) + 1, len(x)))[j], 5)
	y = np.clip(y, VarMin, VarMax)
	return y

def getMutationPopulation(pop = None):
	useRandomSelection = 'Random'
	popm = [complex_data() for each in range(global_configuration.nm)]
	random.shuffle(pop)
	for k in range(0, global_configuration.nm):
		i = random.randint(0, global_configuration.npop-1)
		p = pop[i]
		popm[k].position = Mutate(p.position, global_configuration.mu,
								  global_configuration.varmin, global_configuration.varmax)
		popm[k] = Fitness((popm[k], 1))
	return popm

def getCost(pop, index=0):
    return pop[index].cost

def vector_random_generator(start, end, size):
    result = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            result[i][j] = random_generator(start, end)
    return result

def random_generator(start, end):
    range_ = end - start
    randNum = start + random.random() * range_
    return randNum
