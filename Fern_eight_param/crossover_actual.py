import numpy as np
from complex_data import complex_data
import math
from global_configuration import global_configuration
import random
from Fitness import Fitness

def getCrossOverPopulation(pop = None):
	useRandomSelection = 'Random'
	popc = [[complex_data() for j in range(2)] for i in range(math.ceil(global_configuration.nc/2))]
	random.shuffle(pop)
	for k in range(0, math.ceil(global_configuration.nc/2)):
		#if useRouletteWheelSelection is not None:
			#i1 = RouletteWheelSelection(P)
			#i2 = RouletteWheelSelection(P)
		if useRandomSelection is not None:
			i1 = random.randint(1, global_configuration.npop - 1)
			i2 = i1
			while i2 == i1:
				i2 = random.randint(1, global_configuration.npop - 1)

		p1 = pop[i1]
		p2 = pop[i2]
		print("before cross over:")
		print (p1.position)
		print (p2.position)
		popc[k][0].position, popc[k][1].position = Crossover(p1.position,p2.position,global_configuration.gamma,global_configuration.varmin,global_configuration.varmax)
		#popc[k][0].cost = Fitness(popc[k][0].position, getCost(pop, index = len(pop) - 1))
		#popc[k][1].cost = Fitness(popc[k][1].position,getCost(pop, index = len(pop) - 1))
	flatten_popc = []
	for each_elem in popc:
		for sub_elem in each_elem:
			flatten_popc.append(sub_elem)
	return flatten_popc


def Crossover(x1, x2, gamma, VarMin, VarMax):
	alpha = vector_random_generator(-gamma, 1 + gamma, len(x1))
	y1 = np.dot(alpha, x1) + np.dot((1 - alpha), x2)
	y1 = [each if np.isnan(each) == False else (random.randrange(VarMin * 100, VarMax * 100) / 100) for each in y1]
	oldMin = np.min(y1)
	oldMax = np.max(y1)
	y1 = np.round(VarMin + ((y1 - oldMin)*(VarMax - VarMin))/(oldMax - oldMin), 5)

	y2 = np.dot(alpha, x2) + np.dot((1-alpha), x1)
	y2 = [each if np.isnan(each) == False else (random.randrange(VarMin * 100, VarMax * 100) / 100) for each in y2]
	oldMin = np.min(y2)
	oldMax = np.max(y2)
	y2 = np.round(VarMin + ((y2 - oldMin) * (VarMax - VarMin)) / (oldMax - oldMin), 5)
	y1 = np.clip(y1, VarMin, VarMax)
	y2 = np.clip(y2, VarMin, VarMax)
	print("cross over population: ")
	print(y1)
	print(y2)
	return (y1, y2)

def getCost(pop, index=0):
	return pop[index].cost


def vector_random_generator(start, end, size):
	result = np.zeros((size, size))
	for i in range(0, size):
		for j in range(0, size):
			result[i][j] = random_generator(start, end)
	result = np.round(result, 5)
	print("this is from vector_num_gen", result)
	return result


def random_generator(start, end):
	range_ = end - start
	randNum = start + random.random() * range_
	return randNum

if __name__ == '__main__':
	x2 = [0.21651, 0.35631, 0.36532, 0.35261, 0.40381, 0.36531, 0.30663, 0.27358]
	x1 = [0.50999, 0.39681, 0.30929, 0.37265, 0.26156, 0.34083, 0.29692, 0.312  ]
	Crossover(x1, x2, 0.05, -0.4, 0.4)