import codecs
import logging
import os
import shutil
from collections import defaultdict
# from collections import dequ
# import scipy as sp
# import math
import logger
from Fitness import TBO
from Fitness import Fitness
import random
import numpy as np
# from scipy.stats import norm
# from matplotlib import pyplot as plt
# print(random_generator(-.50,.50))
from complex_data import complex_data
from global_configuration import global_configuration
# from functools import partial
import multiprocessing as mp
import GAHelper
#import timing

useRouletteWheelSelection = None
useTournamentSelection = None
useRandomSelection = None
from plotting_data import experiments
# from numba import jit, cuda
# from random import randint
# from crossover import Crossover
from crossover_multiple_param_range import getCrossOverPopulation
from mutation_multiple_params import getMutationPopulation
from plotting_data import pointGenerationFuncVer2
from plotting_data import pointGenerationFunc
import pylab as pl
import sys
from logger_module import Logger


def random_generator(start, end):
    range_ = end - start
    randNum = start + random.random() * range_
    return randNum


def sortPopulation(pop):


    #######
    pop_ = sorted(pop, key=lambda x: (x.cost), reverse=True)
    for i in pop_:
        print ("Sorted cost: ",  i.cost)
    #print("SORT : Sorted costs for initial population:", pop_.)
    #print("SORT : Sorted costs for initial population:", pop_[1:len(pop_)])
    return pop_[0:len(pop_)]



def findBestSolution(currentBest, candidateBest):
    best = currentBest if currentBest.cost > candidateBest.cost else candidateBest
    return best


def initializePopulation():
    Logger.logger_dev.setLevel(logging.INFO)
    pop = [complex_data() for each in range(global_configuration.npop)]

    for i in range(0, global_configuration.npop):
        result = list()
        for j in range(global_configuration.varsize[0], global_configuration.varsize[1]):
            result.append(random_generator(global_configuration.varmin, global_configuration.varmax))
        pop[i].position = np.array(result)
        pop[i] = Fitness((pop[i], 1))
        #print (pop[i])
        Logger.logger_dev.info("Initial population: Pop " + str(i) + ":" + str(pop[i].position))
        Logger.logger_dev.info("Fitness cost:" + str(pop[i].cost))

    
    return pop


def logPopulationInfo(current_output_dir, it, logger, pop):
	Logger.logger_dev.setLevel(logging.INFO)
	if it == 1:
		for i in range(0, global_configuration.npop):
			Logger.logger_dev.info("Initial population: Pop " + str(i), ":" + str(pop[i].position))
			Logger.logger_dev.info("Fitness cost:" + str(pop[i].cost))
			logger.log_initial_pop(current_output_dir, str(pop[i].position), str(pop[i].cost), str(it))
	else:
			pop = pop.copy()
			Logger.logger_dev.info("Length of pop in next iteration", str(len(pop)))
			for i in range(0, len(pop)):
				Logger.logger_dev.info("Pop " + str(i), ":" + str(pop[i].position))
				Logger.logger_dev.info("Cost for Pop " + str(i), ":" + str(pop[i].cost))
				logger.log_initial_pop(current_output_dir, str(pop[i].position), str(pop[i].cost),str(it))
	return

def createOutputDir(root_output_dir, it):
    if os.path.exists(root_output_dir) is False:
        os.mkdir(root_output_dir)
    current_output_dir = root_output_dir + str(it)
    if os.path.exists(current_output_dir) is True:
        shutil.rmtree(current_output_dir)
    os.mkdir(current_output_dir)
    return current_output_dir


def TransformIntoPixels(X, Y):
    pixels = np.array(list(zip(X, Y))) * 10000
    pixels = np.unique(pixels.astype(int), axis=0).astype("float64") / 10000
    X = pixels[:, 0]
    Y = pixels[:, 1]
    return X, Y
# @jit(target="cuda")
def mainFunc():
    # nfe = np.zeros((global_configuration.maxit, 1))
    BestCostSolution = [0 for i in range(0, global_configuration.maxit + 1)]
    BestCost = [0 for i in range(0, global_configuration.maxit + 1)]
    nfe = [0 for i in range(0, global_configuration.maxit + 1)]
    pool = mp.Pool(10)  # mp.cpu_count())
    large_cost = 50000
    global_index = 1
    Logger.logger_dev.setLevel(logging.INFO)
    curr_wrk_dir = os.getcwd()
    root_output_dir = os.path.join(curr_wrk_dir, 'output')
    if logger.clear_directory(root_output_dir) is False:
        Logger.logger_dev.info('Failed to clear output directory')
        return

    with codecs.open(os.path.join(root_output_dir + "fractal_dimension.txt"), mode="w+",encoding="utf-8") as dimension_writer:
        for it in range(1, global_configuration.maxit+1):
            current_output_dir = createOutputDir(root_output_dir, it)
            Logger.logger_dev.info("current output_dir:" + current_output_dir)
            Logger.logger_dev.info("ITERATION" + str(it) )
            Logger.logger_dev.info("Initial population for iteration " + str(it)+ ":")
            pop = initializePopulation()
            logPopulationInfo(current_output_dir, it, logger, pop)
            # sort the population according to cost
            sortedPop = sortPopulation(pop)
            for i in range(0, len(sortedPop)):
                logger.log_sorted_pop(current_output_dir, str(sortedPop[i].position), str(sortedPop[i].cost), str(it))
            Logger.logger_dev.info("Select the top npop best cost population")
            pop = sortedPop[:len(sortedPop)]
            popc = getCrossOverPopulation(pop=pop)
            for i in range(0, len(popc)):
                logger.log_crossover_pop(current_output_dir, str(popc[i].position), str(popc[i].cost), str(it))
            Logger.logger_dev.info("Number of Crossed over children:", len(popc))
            popm = getMutationPopulation(pop=pop)
            for i in range(0, len(popm)):
                logger.log_mutation_pop(current_output_dir, str(popm[i].position), str(popm[i].cost), str(it))
            Logger.logger_dev.info("Number of Mutated children:" + str(len(popm)))
            pop = np.concatenate((popc, popm))  # popc.flatten()
            Logger.logger_dev.info("Total number of final chromosomes in iteration  " +str(it) + "is"+ str(len(pop))+
                                   " Final population from iteration " + str(it)+ ":")
            for i in range (0, len(pop)):
                logger.log_final_pop(current_output_dir, str(pop[i].position), str(pop[i].cost), str(it))
                Logger.logger_dev.info("final_pop_" + str(i) + str(pop[i].position))
            Logger.logger_dev.info("right before submission to pool")
            pop = [each_pop for each_pop in pop if len(each_pop.position) == 12]
            pop = pool.map(Fitness, [(datum, global_index + index) for index, datum in enumerate(pop)])
            # sort the population according to cost
            sorted_finalPop = sortPopulation(pop)
            Logger.logger_dev.info("Best Fitness cost in iteration:" + str(sorted_finalPop[0].cost))
            Logger.logger_dev.info("Select the top npop best cost population")
            #pop = sortedPop[:global_configuration.npop] #number of population = as assigned before
            pop = sorted_finalPop[:len(pop)] #keeping number of population as the total
            BestSolution = pop[0]  # findBestSolution(BestSolution, pop[0])
            Logger.logger_dev.info("At iteration " + str(it) + " Results are: ")
            Logger.logger_dev.info("All the fitness cost for this iteratration:")
            Logger.logger_dev.info(str([pop[i].cost for i in range(len(pop))]))
            Logger.logger_dev.info("Best Point: " + str([each_point for each_point in BestSolution.position]))
            Logger.logger_dev.info("Best Cost: " + str(BestSolution.cost))
            BestCostSolution[it] = BestSolution
            BestCost[it] = BestSolution.cost
            global_index += 100
            dimension_writer.write("Best fractal dimension for iteration " +str(it) + " is: "  + str(BestSolution.cost) + '\n')
            X, Y = pointGenerationFuncVer2(BestSolution.position)
            X, Y = TransformIntoPixels(X, Y)
            experiments(X, Y, "Fern_" + str(it) + ".png")
            nfe[it] = it
        pool.close()
        pool.join()
    print('After the final Iteration :' + str(it) + ': NFE = ' + str(nfe[it]) + ', Best Cost= ' + str(BestCost[it]))

        
def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def RouletteWheelSelection(P):
    c = np.cumsum(P)
    r = [random.random() for i in range(0, len(c))]
    i = indices(r <= c, lambda x: x != False)[0]
    return i

def TournamentSelection(pop, m):
    nPop = len(pop)
    S = random.randint(0, nPop)
    spop = pop[S]
    scosts = spop.cost
    j = min(scosts)
    i = S[j]
    return i

if __name__ == '__main__':
    mainFunc()
