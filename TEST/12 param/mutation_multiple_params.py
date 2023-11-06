import sys

import numpy as np
from complex_data import complex_data
import math
from global_configuration import global_configuration
import random
from Fitness import Fitness
from scipy.stats import norm

from global_configuration import global_configuration


def Mutate(x, mu, VarMin, VarMax):
    nVar = len(x[0:8])
    nmu = math.ceil(mu * nVar)
    np.random.seed(1)
    j = random.randint(min(nVar, nmu), max(nVar, nmu))
    nVar = len(x[8:10])
    nmu = math.ceil(mu * nVar)
    j1 = random.randint(min(nVar, nmu), max(nVar, nmu))
    sigma = 0.1 * (VarMax - VarMin)
    sigma2 = 0.1 * (global_configuration.varmax_param_1_2 - global_configuration.varmin_param_1_2)
    sigma3 = 0.1 * (global_configuration.varmax_param_3_4 - global_configuration.varmin_param_3_4)
    y1 = np.round(x[0:8] + sigma * norm.ppf(np.random.rand(len(x[0:8]) + 1, len(x[0:8])))[j], 5)
    y2 = np.round(x[8:10] + sigma2 * norm.ppf(np.random.rand(len(x[8:10]) + 1, len(x[8:10])))[j1], 5)
    y3 = np.round(x[10:12] + sigma3 * norm.ppf(np.random.rand(len(x[10:12]) + 1, len(x[10:12])))[j1], 5)
    print("Mutated population: ")
    print(y1)
    print(y2)
    y1 = np.clip(y1, VarMin, VarMax)
    y2 = np.clip(y2, global_configuration.varmin_param_1_2, global_configuration.varmax_param_1_2)
    y3 = np.clip(y3, global_configuration.varmin_param_3_4, global_configuration.varmax_param_3_4)

    return np.concatenate((np.concatenate((y1 , y2), axis=0), y3), axis=0)


def getMutationPopulation(pop=None):
	useRandomSelection = 'Random'
	popm = [complex_data() for each in range(global_configuration.nm)]
	random.shuffle(pop)
	for k in range(0, global_configuration.nm):
		i = random.randint(0, global_configuration.npop-1)
		p = pop[i]
		popm[k].position = Mutate(p.position, global_configuration.mu, global_configuration.varmin, global_configuration.varmax)
		#popm[k].cost = Fitness(popm[k].position, getCost(pop, index = len(pop) - 1))
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
