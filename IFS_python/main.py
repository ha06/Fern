# importing necessary modules
import matplotlib.pyplot as plt
from random import randint

# initializing the list
x = []
y = []

# setting first element to 0
x.append(0)
y.append(0)

current = 0

for i in range(1, 6000000):

    #generates a random integer between 1 and 100
    z = randint(1, 100)

    #the x and y coordinates of the equations are appended in the lists respectively.

    #for the probability 0.01
    if z == 1:
        x.append(0)
        y.append(0.16 * (y[current]))

        #for the probability 0.85
    # if z >= 2 and z <= 86:
        # x.append(0.85 * (x[current]) + 0.04 * (y[current]))
        # y.append(-0.04 * (x[current]) + 0.85 * (y[current]) + 1.6)
    #
    if z >= 2 and z <= 86:
        x.append(0.33119299 * (x[current]) + 0.4 * (y[current]))
        y.append(-0.08 * (x[current]) + 0.36 * (y[current]) + 1.6)

    #for the probability 0.07
    if z >= 87 and z <= 93:
        x.append(0.2 * (x[current]) - 0.26 * (y[current]))
        y.append(0.23 * (x[current]) + 0.22 * (y[current]) + 1.6)

       # for the probability 0.07
    if z >= 94 and z <= 100:
        x.append(-0.15 * (x[current]) + 0.28 * (y[current]))
        y.append(0.26 * (x[current]) + 0.24 * (y[current]) + 0.44)



    current = current + 1

# for i in range(1, 200000):
#
#     # generates a random integer between 1 and 100
#     z = randint(1, 100)
#
#         # the x and y coordinates of the equations
#         # are appended in the lists respectively.
#     print("Iteration number: " + str(i))
#     print("Z values is: " + str(z))

    # for the probability 0.01
    # if z >= 1 and z<=10:
    #     x.append(0.14 * (x[current]) + 0.01 * (y[current]) - 0.08)
    #     y.append(0 + 0.51 * (y[current]) - 1.31)
    #
    #         # for the probability 0.85
    # if z >= 11 and z <= 45:
    #     x.append(0.43 * (x[current]) + 0.52 * (y[current]) + 1.49)
    #     y.append(-0.45 * (x[current]) + 0.50 * (y[current]) -0.75)
    #
    #         # for the probability 0.07
    # if z >= 46 and z <= 80:
    #     x.append(0.45 * (x[current]) - 0.49 * (y[current]) - 1.62)
    #     y.append(0.47 * (x[current]) + 0.47 * (y[current]) - 0.74)
    #
    #         # for the probability 0.07
    # if z >= 81 and z <= 100:
    #     x.append(-0.49 * (x[current]) + 0.02)
    #     y.append(0+ 0.51 * (y[current]) + 1.62)
    #
    # current = current + 1




    #### Maple leaf
    # if z >= 1 and z<=10:
    #     x.append(0.1 * (x[current]) + 0.02 * (y[current]) - 0.08)
    #     y.append(0 - 0.51 * (y[current]) - 1.31)
    #
    #         # for the probability 0.85
    # if z >= 11 and z <= 45:
    #     x.append(0.33 * (x[current]) + 0.42 * (y[current]) + 1.49)
    #     y.append(-0.45 * (x[current]) + 0.30 * (y[current]) -0.75)
    #
    #         # for the probability 0.07
    # if z >= 46 and z <= 80:
    #     x.append(0.45 * (x[current]) - 0.45 * (y[current]) - 1.62)
    #     y.append(0.47 * (x[current]) + 0.47 * (y[current]) - 0.74)
    #
    #         # for the probability 0.07
    # if z >= 81 and z <= 100:
    #     x.append(-0.50 * (x[current]) + 0.02)
    #    # y.append(0+ 0.51 * (y[current]) + 1.62)
    #     y.append(0 + 0.51 * (y[current]) + 1.62)
    #
    # current = current + 1


plt.scatter(x, y, s=0.2, edgecolor='green')
plt.savefig('fern.png')
plt.show()

