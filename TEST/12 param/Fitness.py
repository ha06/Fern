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
NFE = 0
#from sklearn.linear_model import SGDRegressor
#from sklearn.linear_model.logistic import _logistic_loss
##################################

def get_rows():
    return

def iter_minibatches(X, y, y1, chunksize=10000):
    # Provide chunks one by one
    chunkstartmarker = 0
    numtrainingpoints = len(X)
    while chunkstartmarker < numtrainingpoints:
        # chunkrows = range(chunkstartmarker,chunkstartmarker+chunksize)
        X_chunk, y_chunk, y1_chunk = X[chunkstartmarker:chunkstartmarker + chunksize],\
                                     y[chunkstartmarker: chunkstartmarker + chunksize], \
                                     y1[chunkstartmarker: chunkstartmarker + chunksize]
        yield X_chunk, y_chunk, y1_chunk
        chunkstartmarker += chunksize

###########################################################
def random_generator(start, end):
    range_ = end - start
    randNum = start + random.random() * range_
    return randNum

def TBO(w,index):
    iterations = global_configuration.iterations
    keyval = defaultdict(int)
    #scales = np.logspace(-8, 1., num=50, endpoint=True, base=2)
    #you are creating 100 points in logspace, reduce korte hobe. 50??
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
            #x.append(0.85 * (x[current]) + 0.04 * (y[current]))
            #y.append(-0.04 * (x[current]) + 0.85 * (y[current]) + 1.6)
            print("pop width: " + str(w))
            print("current pop:" + str(current))
            print("lenght of iter: " + str(len(x)) + " "+ str(len(y)))
            x.append(w[10] * (x[current]) + w[8] * (y[current]))
            y.append(w[9] * (x[current]) + w[11] * (y[current]) + 1.6)
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
    return cost

def Fitness(args):
    x_pop, index = args
    cost_val = TBO(x_pop.position, index)
    x_pop.cost= -1*cost_val if cost_val <= 0 else cost_val
    return x_pop

if __name__=='__main__':
    pass
    #print(Fitness())
