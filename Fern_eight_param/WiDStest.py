#importing necessary modules
import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from random import randint
from random import random
import random
import numpy as np
import os
from global_configuration import global_configuration
from matplotlib.ticker import FormatStrFormatter

def random_generator(start, end):
    range_ = end - start
    randNum = start + random.random() * range_
    return randNum

def pointGenerationFunc(w):
    iterations = global_configuration.iterations
    A1 = np.array([[0, 0], [0, 0.17]])
    A2 = np.array([[0.85, 0.04], [-0.04, 0.85]])
    A3 = np.array([[w[0], w[1]], [w[2], w[3]]])
    A4 = np.array([[w[4], w[5]], [w[6], w[7]]])
    t1 = np.array([[0], [0]])
    t2 = np.array([[0], [1.6]])
    t3 = np.array([[0], [1.6]])
    t4 = np.array([[0], [0.44]])
    p1 = 0.01
    p2 = 0.85
    p3 = 0.07
    p4 = 0.07
    x = [0]
    y = [0]
    y1 = [0]
    v = np.array([[0], [0]])

    for n in range(1, iterations):
        k = random_generator(0, 1)
        if (k < p1):
            v = np.dot(A1, v) + t1
        elif (k < p1 + p2):
            v = np.dot(A2, v) + t2
        elif (k < p1 + p2 + p3):
            v = np.dot(A3, v) + t3
        else:
            v = np.dot(A4, v) + t4
        # now, go back and define your (x,y) point as elements of the vector v
        x.append(v[0][0])
        y.append(v[1][0])
    return x, y

def pointGenerationFuncVer2(w):
    x = []
    y = []
    x.append(0)
    y.append(0)
    current = 0
    #for i in range(1, global_configuration.iterations):
    for i in range(1, 1000000):
        # generates a random integer between 1 and 100
        z = randint(1, 100)
        # the x and y coordinates of the equations are appended in the lists respectively.

        # for the probability 0.01
        #if z == 1:
        if z >= 1 and z <= 3:
            x.append(0)
            y.append(w[3] * (y[current]))

        # for the probability 0.85
        #if z >= 2 and z <= 86:
        if z >= 4 and z <= 80:
            x.append(w[4] * (x[current]) + w[5] * (y[current]))
            y.append(w[6] * (x[current]) + w[7] * (y[current]) + 1.6)
            # x.append(w[0] * (x[current]) + w[1] * (y[current]))
            # y.append(w[2] * (x[current]) + w[3] * (y[current]) + 1.6)

        # for the probability 0.07
        #if z >= 87 and z <= 93:
        if z >= 81 and z <= 91:
            x.append(w[8] * (x[current]) - w[8] * (y[current]))
            y.append(w[10] * (x[current]) + w[11] * (y[current]) + 1.6)

        # for the probability 0.07
        #if z >= 94 and z <= 100:
        if z >= 92 and z <= 100:
            x.append(w[12] * (x[current]) + w[13] * (y[current]))
            y.append(w[14] * (x[current]) + w[15] * (y[current]) + 0.44)
        current = current + 1
    #print("TEST", x,y)
    return (x, y)

def experiments(X, Y,file_name=None):
    print('trying to plot')
    plt.clf()
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_ticks(np.arange(-4, 12, 1))
    plt.scatter(X, Y, s=0.2, edgecolor='green')
    plt.savefig(os.getcwd() + "/plot_outputs/" + 'TRIAL_fern.png') if file_name is None else plt.savefig(os.path.join(os.getcwd(), "plot_outputs/") + file_name)
    plt.show()

if __name__ == '__main__':
    w4=  [0.19876097, -0.03807146, 0.26342428, 0.31630758, -0.08032394, 0.1001038, 0.25267275, -0.18849974]
    #w = [0.2, -0.26, 0.23, 0.22, -0.15, 0.28, 0.26, 0.24]
    w3 = [-0.21307426, - 0.2047206, - 0.1930553, - 0.14540351,  0.22778541,  0.14486501,  -0.29428478, -0.1343652 ]
    #w = [-0.06686571, -0.27881328, -0.34893844, -0.02336819,  0.1530533,  -0.12028196,  -0.29601663, -0.25442507]
    w1 = [0.02402978, -0.20749615, -0.27449872, -0.04912554, 0.19085048, -0.12817851,  -0.01902274, -0.20727737]
    w2 = [-0.06686571, -0.27881328, -0.34893844, -0.02336819, 0.1530533, -0.12028196, -0.29601663, -0.25442507]
    w5 = [-0.27254205, 0.02593793, 0.22694705, -0.03612513, 0.20730881, 0.02912449, 0.05677947, 0.01115072]
    w6 = [-0.12071485, -0.07843765, -0.00840219, -0.04626357, -0.20122522, -0.20728113, -0.15799326, -0.17638013]

    w = [0.34971313, 0.35, 0.35, 0.12301834, 0.0884259,  0.35, 0., 0.35, -0.12071485, -0.07843765, -0.00840219, -0.04626357, -0.20122522, -0.20728113, -0.15799326, -0.17638013]


    X,Y = pointGenerationFuncVer2(w)
    experiments(X, Y)
