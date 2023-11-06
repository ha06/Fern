#from collections import defaultdict 
#import math
#import pylab as pl
#import scipy as sp
#from matplotlib import pyplot as plt
import numpy as np
import random
from global_configuration import global_configuration
from housdroff import cost_function
from collections import defaultdict
from complex_data import complex_data
NFE = 0
#from sklearn.linear_model import SGDRegressor
#from sklearn.linear_model.logistic import _logistic_loss
##################################


def random_generator(start, end):
    range_ = end - start
    randNum = start + random.random() * range_
    return randNum

def TBO(w, index):
    iterations = global_configuration.iterations
    scales = np.logspace(-4., 1, num=50, endpoint=True, base=2)


    x = [0]
    y = [0]

    current = 0
    for n in range(1, iterations):
        k = random.randint(1, 100)
        # k = random_generator(0,1)
        if k == 1:  # if (k <= p1):
        #if k == 10:
            # v= np.dot(A1 , v) + t1
            x.append(0)
            y.append(0.16 * (y[current]))
        elif k >= 2 and k <= 86:  # (k < p1+p2):
            # v= np.dot(A2 , v) + t2
            x.append(0.85 * (x[current]) + 0.04 * (y[current]))
            y.append(-0.04 * (x[current]) + 0.85 * (y[current]) + 1.6)
        elif k >= 87 and k <= 93:  # ( k < p1 + p2 + p3):
            # v= np.dot(A3 , v ) + t3
            x.append(w[0] * (x[current]) - w[1] * (y[current]))
            y.append(w[2] * (x[current]) + w[3] * (y[current]) + 1.6)
        elif k >= 94 and k <= 100:
            # v= np.dot(A4 , v) + t4
            x.append(w[4] * (x[current]) + w[5] * (y[current]))
            y.append(w[6] * (x[current]) + w[7] * (y[current]) + 0.44)
        else:
            pass

        current += 1
    cost = cost_function(x, y, scales, index)
    print("cost before eq.", cost)
    cost = -1 * cost if cost <= 0 else cost
    print("cost after eq:", cost)
    print(w)
    print("cost:", cost)
    return cost


def Fitness(args):
    x_pop, index = args
    cost_val = TBO(x_pop.position, index)
    print(x_pop)
    print("cost before eq.", cost_val)
    x_pop.cost= -1*cost_val if cost_val <= 0 else cost_val
    print("cost after eq:", x_pop.cost)
    return x_pop

if __name__=='__main__':
    #pass
    #print(Fitness())
    w = (complex_data ())
    w.position = [0., 0.31255727, 0.5, 0.5, 0.29578599, 0.5, 0.5, 0.08121665]
    #w.position = [-0.32250142,  0.47830325,  0.3237394,   0.091068,    0.21604983,  0.12351257, 0.40637589,  0.29406173]
    w.cost = 500
    TBO(w,1)
    #Fitness(w.position, 1)