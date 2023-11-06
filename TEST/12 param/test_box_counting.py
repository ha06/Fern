import traceback
import pylab as pl
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from random import randint
from random import random
import random
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from complex_data import complex_data
import os
def random_generator(start, end):
    range_ = end - start
    randNum = start + random.random() * range_
    return randNum

def pointGenerationFuncVer2(w):
    scales = np.logspace(-4., 0.9, num= 100, endpoint=True, base=2)
    index = 1
    x = []
    y = []
    x.append(0)
    y.append(0)
    current = 0
    #for i in range(1, global_configuration.iterations):
    for i in range(1, 50000):
        # generates a random integer between 1 and 100
        z = randint(1, 100)
        # the x and y coordinates of the equations
        # are appended in the lists respectively.

        # for the probability 0.01
        if z == 1:
            x.append(0)
            y.append(0.16 * (y[current]))

        # for the probability 0.85
        if z >= 2 and z <= 86:
            x.append(0.85 * (x[current]) + 0.04 * (y[current]))
            y.append(-0.04 * (x[current]) + 0.85 * (y[current]) + 1.6)

        # for the probability 0.07
        if z >= 87 and z <= 93:
            x.append(w[0] * (x[current]) - w[1] * (y[current]))
            y.append(w[2] * (x[current]) + w[3] * (y[current]) + 1.6)

        # for the probability 0.07
        if z >= 94 and z <= 100:
            x.append(w[4] * (x[current]) + w[5] * (y[current]))
            y.append(w[6] * (x[current]) + w[7] * (y[current]) + 0.44)
        current = current + 1

    cost = cost_function(x, y, scales, index)
    cost = -1 * cost if cost <= 0 else cost
    print(w)
    print("cost:", cost)
    experiments(x,y)
    return cost
    return (x, y)

def experiments(X, Y,file_name=None):
    print('trying to plot')
    plt.clf()
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_ticks(np.arange(-3, 12, 1))
    plt.scatter(X, Y, s=0.2, edgecolor='green')
    plt.savefig(os.getcwd() + "/plot_outputs/" + 'TRIAL_fern.png') if file_name is None else plt.savefig(os.path.join(os.getcwd(), "plot_outputs/") + file_name)
    plt.show()


param_dist_sgd = {'clf__loss': ['huber'],
                  'clf__alpha': np.linspace(0.15, 0.35),
                  'clf__penalty': ['l1', 'l2', 'elasticnet'],
                  }


def cost_function(x=[], y=[], scales=[], index = 1):
    global param_dist_sgd
    # y1 = np.array(y1)
    x = np.array(x)
    y = np.array(y)
    print("value of x:", x)
    print("value of y:", y)
    x = x - np.min(x)
    y = y - np.min(y)
    print("value of x1:", x)
    print("value of y1:", y)
    # I use the factor 1.1 to ensure that, after dividing, all coordinates lie strictly below 1
    # scale = np.max([np.max(x),np.max(y),np.max(y1)])*1.1
    #x = x*(1./scale)
    #y = y*(1./scale)
    Lx = max(x)
    Ly = max(y)
    pixels = np.array(list(zip(x, y))) * 10000
    pixels = np.unique(pixels.astype(int), axis=0).astype("float64") / 10000
    # lr = global_configuration.lr
    models = [LinearRegression(), GradientBoostingRegressor(), SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.1,
                                                                         penalty="elasticnet", fit_intercept=True,
                                                                         loss='huber')]
    lr_model = Pipeline([('scl', StandardScaler()), ('clf', models[2])])

    #sgd_randomized_pipe = RandomizedSearchCV(estimator=lr_model, param_distributions=param_dist_sgd,
    #                                         cv=3, n_iter=1000, n_jobs=-1)
    #sgd_randomized_pipe = GridSearchCV(estimator=lr_model, param_grid=[param_dist_sgd],
    #                                   cv=3, n_jobs=-1)
    # lr = make_pipeline(StandardScaler(), lr_model)
    Ns = []
    total = 0
    m = -1
    # scales = scales[scales != 1.0]
    scales = scales[scales != None]
    # looping over several scales
    try:
        scale_ = 0
        for scale in scales:
            # print("======= Scale :", scale)
            # computing the histogram
            scale_ = scale
            H, edges = np.histogramdd(pixels, bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
            # Ns.append(np.sum(H > 0))
            Ns.append(np.sum(H > 0) if np.sum(H > 0) > 0 else 1)
        # if(np.isfinite(np.log(Ns[len(Ns) - 1])) == True and np.log(Ns[len(Ns) - 1])!= np.nan):
        # scale = 1./scale if scale < 1.0 else scale
        # total += np.fabs(np.log(Ns[len(Ns) - 1])/(np.log(2*scale)+1.))
        # total += np.fabs(np.ma.masked_invalid(np.log(Ns[len(Ns) - 1])))/np.log(2*scale)
        X = np.array(np.log(1. / scales))
        Y = np.array(np.log(Ns))


        model = lr_model.fit(X.reshape(-1, 1), Y)  # (np.hstack((X.reshape(-1,1 ), Y.reshape(-1,1 ))), Y1)
        # print("model coefficient" + str(global_configuration.lr_model.coef_))
        # m = _logistic_loss(global_configuration.lr_model.coef_, X.reshape(-1, 1), Y,1 / global_configuration.lr_model.C)
    # print("Fractal dimension (Box counting): " + str(-global_configuration.lr_model.coef_))
    # print("current cost calculated: " + str(m))
    # linear fit, polynomial of degree 1
    # coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
    except ValueError as ve:
        print("scale is: ", str(scale_))
        traceback.print_exc()
        print("Its a value error")
        coeffs = [-1]
        return 500
    pl.plot(X, Y, 'o', mfc='none')
    # pl.plot(np.log(scales), np.polyval(coeffs, np.log(scales)))
    pl.xlabel('log 1/$\epsilon$')
    pl.ylabel('log N')
    #pl.plot(X, lr_model.named_steps['clf'].predict(
    #            lr_model.named_steps['scl'].transform(X.reshape(-1, 1))),color='r')
    #pl.show()
    pl.savefig('Hausdorff_dimension' + str(index) +'.png')
    # print("The Hausdorff dimension is", -coeffs[0])  # the fractal dimension is the OPPOSITE of the fitting coefficient
    np.savetxt("scaling.txt", list(zip(scales, Ns)))
    #m = sgd_randomized_pipe.best_estimator_.named_steps['clf'].coef_
    m = lr_model.named_steps['clf'].coef_
    print ("TEST HD", m)
    return -m[0]  # -lr_model.coef_#global_configuration.lr_model.coef_

if __name__=='__main__':
    w1 = [0.45063, -0.06906, 0.2725, -0.1156, -0.48707, -0.40194, 0.2323, -0.15981]
    w2 = [0.35, 0.34642403, 0.221624, 0.30956092, 0.35, 0.18460954, 0.0, 0.35]
    w3 = [0.2, -0.26, 0.23, 0.22,-0.15, 0.28, 0.26, 0.24]
    w = [ 0.346, 0.3780, 0.21326,  0.22668, -0.15528,  0.35, 0.25963,  0.29807]




    pointGenerationFuncVer2(w)
    #experiments(X, Y)

