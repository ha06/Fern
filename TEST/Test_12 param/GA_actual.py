import codecs
import os
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
from crossover import getCrossOverPopulation
from mutation import getMutationPopulation
from plotting_data import pointGenerationFuncVer2
from plotting_data import pointGenerationFunc
import pylab as pl
import sys

curr_wrk_dir = os.getcwd()
# print(curr_wrk_dir)
# Here, we assigin a variable to the output directory
root_output_dir = curr_wrk_dir + '/output/'

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


def Initialization():
    # mutation rate
    global useRouletteWheelSelection
    global useRandomSelection
    ANSWER = 'Random'  # input('Choose selection method: , Genetic Algorithm ,...  Roulette Wheel , Tournament , Random , Roulette Wheel')
    useRouletteWheelSelection = 'Roulette Wheel' if (str.lower(ANSWER) == str.lower('Roulette Wheel')) else None
    useTournamentSelection = 'Tournament' if (str.lower(ANSWER) == str.lower('Tournament')) else None
    useRandomSelection = 'Random' if (str.lower(ANSWER) == str.lower('Random')) else None

    if useRouletteWheelSelection is not None:
        beta = 8  # Selection Pressure

    if useTournamentSelection is not None:
        TournamentSize = 3  # Tournamnet Size

    pop = [complex_data() for each in range(global_configuration.npop)]
    costS = list()

    for i in range(0, global_configuration.npop):
        result = list()
        for j in range(global_configuration.varsize[0], global_configuration.varsize[1]):
            result.append(random_generator(global_configuration.varmin, global_configuration.varmax))
        pop[i].position = np.array(result)
        pop[i] = Fitness((pop[i], 1))
        #print (pop[i])
        print("Initial population: Pop " + str(i), ":" + str(pop[i].position))
        print("Fitness cost:" + str(pop[i].cost))

        #logger.log_initial_pop(root_output_dir, str(pop[i].position), str(pop[i].cost), str(1))
        #sorted_pop = sortPopulation(pop)
        #print("Sorted costs for initial population:", sorted_pop[i].cost)

    sorted_pop = sortPopulation(pop)
    print("Length of Sorted population:", len(sorted_pop))

    BestSolution = sorted_pop[0]
    WorstSolution = sorted_pop[len(sorted_pop) - 1]
    print("Best cost is: " + str(BestSolution.cost))
    print("Get Worst Cost: " + str(WorstSolution.cost))
    return (pop, WorstSolution)

# @jit(target="cuda")
def mainFunc(pop):
    # nfe = np.zeros((global_configuration.maxit, 1))
    BestCostSolution = [0 for i in range(0, global_configuration.maxit + 1)]
    BestCost = [0 for i in range(0, global_configuration.maxit + 1)]
    nfe = [0 for i in range(0, global_configuration.maxit + 1)]
    pool = mp.Pool(10)  # mp.cpu_count())
    large_cost = 50000
    global_index = 1
    try:
        if logger.clear_directory(root_output_dir):

            with codecs.open(os.path.join(root_output_dir + "fractal_dimension.txt"), mode="w+",encoding="utf-8") as dimension_writer:
                for it in range(1, global_configuration.maxit+1):
                    # P = np.exp(-global_configuration.beta * costS / getCost(pop, index=len(pop) - 1))
                    # P = P / np.sum(P)
                    os.mkdir(root_output_dir + str(it))
                    current_output_dir = root_output_dir + str(it) + '/'
                    print("current output_dir:", current_output_dir)
                    #prev_pop = pop.copy()
                    print("ITERATION" + str(it) )
                    print ("Initial population for iteration " + str(it), ":")

                    if it == 1:
                        #pop = Initialization()
                        #for i in range(0, len(prev_pop)):
                        for i in range(0, global_configuration.npop):
                            prev_pop = pop.copy()
                            #prev_pop = pop
                            print("Initial population: Pop " + str(i), ":" + str(prev_pop[i].position))
                            print("Cost for Pop " + str(i), ":" + str(prev_pop[i].cost))
                            logger.log_initial_pop(current_output_dir, str(prev_pop[i].position), str(prev_pop[i].cost), str(it))


                    #if it >= 2 and BestCost[it - 2] > BestCost[it - 1]:
                    else:
                        pop = pop.copy()
                        print("Length of pop in next iteration", str(len(pop)))
                        for i in range(0, len(pop)):
                            print("Pop " + str(i), ":" + str(pop[i].position))
                            print("Cost for Pop " + str(i), ":" + str(pop[i].cost))
                            logger.log_initial_pop(current_output_dir, str(pop[i].position), str(pop[i].cost),str(it))

                    # sort the population according to cost
                    sortedPop = sortPopulation(pop)
                    for i in range(0, len(sortedPop)):
                        logger.log_sorted_pop(current_output_dir, str(sortedPop[i].position), str(sortedPop[i].cost), str(it))

                    print("Elite population from iteration :" + str(it) , "is:" + str(sortedPop[0].cost))
                    print("Select the top npop best cost population")
                    pop = sortedPop[:len(sortedPop)]

                    popc = getCrossOverPopulation(pop=pop)
                    #print("IN GA after CO:", str(str([(each_elem.position, each_elem.cost) for each_elem in popc])))

                    for i in range(0, len(popc)):
                        logger.log_crossover_pop(current_output_dir, str(popc[i].position), str(popc[i].cost), str(it))

                    print("Number of Crossed over children:", len(popc))

                    popm = getMutationPopulation(pop=pop)
                    for i in range(0, len(popm)):
                        logger.log_mutation_pop(current_output_dir, str(popm[i].position), str(popm[i].cost), str(it))

                    print("Number of Mutated children:", len(popm))
                    # merge two different types of mutation
                    pop = np.concatenate((popc, popm))  # popc.flatten()
                    print("Total number of final chromosomes in iteration  " +str(it) , "is", len(pop), " Final population from iteration " + str(it), ":")
                    for i in range (0, len(pop)):
                        logger.log_final_pop(current_output_dir, str(pop[i].position), str(pop[i].cost), str(it))
                        print("final_pop_" +str(i), pop[i].position)
                    print("right before submission to pool")
                    pop = pool.map(Fitness, [(datum, global_index + index) for index, datum in enumerate(pop)])
                    # sort the population according to cost
                    sorted_finalPop = sortPopulation(pop)
                    print("Best Fitness cost in iteration:" + str(sorted_finalPop[0].cost))
                    print("Select the top npop best cost population")
                    #pop = sortedPop[:global_configuration.npop] #number of population = as assigned before
                    pop = sorted_finalPop[:len(pop)] #keeping number of population as the total
                    #pop = sorted_finalPop[:110] #keeping number of population as 12
                    #find the best solution
                    BestSolution = pop[0]  # findBestSolution(BestSolution, pop[0])
                    print("At iteration " + str(it) + " Results are: ")
                    print("All the fitness cost for this iteratration:")
                    #print([pop[i].cost for i in range(global_configuration.npop)])
                    print([pop[i].cost for i in range(len(pop))])
                    print("Best Point: " + str([each_point for each_point in BestSolution.position]))
                    print("Best Cost: " + str(BestSolution.cost))
                    BestCostSolution[it] = BestSolution
                    BestCost[it] = BestSolution.cost
                    global_index += 100
                    dimension_writer.write("Best fractal dimension for iteration " +str(it) + " is: "  + str(BestSolution.cost) + '\n')
                    X, Y = pointGenerationFuncVer2(BestSolution.position)
                    pixels = np.array(list(zip(X, Y))) * 10000
                    pixels = np.unique(pixels.astype(int), axis=0).astype("float64") / 10000
                    X = pixels[:, 0]
                    Y = pixels[:, 1]
                    experiments(X, Y, "Fern_" + str(it) + ".png")
                    nfe[it] = it
            pool.close()
            pool.join()
    # pl.plot(np.array([i for i in range(global_configuration.maxit)]), np.array(BestCost), 'o', mfc='none')
    # pl.plot(np.log(scales), np.polyval(coeffs, np.log(scales)))
    # pl.xlabel('Iterations')
    # pl.ylabel('Fractal Dimension')
    # pl.plot(X, lr.predict(X.reshape(-1,1)),color='k')
    # pl.show()
    # pl.savefig('Hausdorff_dimension.png')
            print('After the final Iteration :' + str(it) + ': NFE = ' + str(nfe[it]) + ', Best Cost= ' + str(BestCost[it]))
    # boxCountingPlot(BestSolution.position)
            return BestCostSolution[global_configuration.maxit - 1]
        else:
            print('Failed to clear output directory')
    except Exception as e:
        print(e)
        print("Error in running the GA")

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

def multiProcessingOps():

    '''
    try:
        taskid =os.environ["SGE_TASK_ID"]
        output_filename = "fern_output.{0}".format(taskid)
    except KeyError:
        print("Could not read SGE_TASK_ID")
        sys.exit(0)
    '''

    print("START?")
    #output_filename = "fern_output.{0}".format('png')
    output_filename = "doubt".format('png')

    pop, worstSolution = Initialization()
    #bestParameters = mainFunc(pop, worstSolution)
    mainFunc(pop)
    # print("Best Parameter is: " + "".join(str(bestParameters.position)))
    # print("Best cost is: " + str(bestParameters.cost))
    # X, Y = pointGenerationFuncVer2(bestParameters.position)
    # experiments(X, Y, output_filename)

if __name__ == '__main__':
    #Initialization()
    #pop = complex_data()
    #mainFunc(pop)
    multiProcessingOps()
