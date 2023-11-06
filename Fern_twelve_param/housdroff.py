from turtledemo.chaos import line

import numpy as np
# import sys
# import scipy as sp
import traceback
# from sklearn.linear_model import SGDRegressor
import pylab as pl
#from sklearn.linear_model.logistic import _logistic_loss
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold

from global_configuration import global_configuration
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import sys


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


param_dist_sgd = {'clf__loss': ['huber'],
                  'clf__alpha': np.linspace(0.15, 0.35),
                  'clf__penalty': ['l1', 'l2', 'elasticnet'],
                  }


def cost_function(x=[], y=[], scales=[], index = 1):
    global param_dist_sgd
    # y1 = np.array(y1)
    x = np.array(x)
    y = np.array(y)
    x = x - np.min(x)
    y = y - np.min(y)
    # I use the factor 1.1 to ensure that, after dividing, all coordinates lie strictly below 1
    # scale = np.max([np.max(x),np.max(y),np.max(y1)])*1.1
    # x = x*(1./scale)
    # y = y*(1./scale)
    Lx = max(x)
    Ly = max(y)
    pixels = np.array(list(zip(x, y))) * 10000
    pixels = np.unique(pixels.astype(int), axis=0).astype("float64") / 10000
    # lr = global_configuration.lr
    models = [LinearRegression(), GradientBoostingRegressor(), SGDRegressor(max_iter=1000, tol=1e-4, eta0=0.1,
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
            # computing the histogra
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
        # plt.scatter(X,Y,s=0.2,edgecolor='green')

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
        return 0
    pl.plot(X, Y, 'o', mfc='none')
    # pl.plot(np.log(scales), np.polyval(coeffs, np.log(scales)))
    pl.xlabel('log 1/$\epsilon$')
    pl.ylabel('log N')
    #pl.plot(X, lr_model.named_steps['clf'].predict(
    #            lr_model.named_steps['scl'].transform(X.reshape(-1, 1))),color='r')
    #pl.show()
    #pl.savefig('Hausdorff_dimension' + str(index) +'.png')
    # print("The Hausdorff dimension is", -coeffs[0])  # the fractal dimension is the OPPOSITE of the fitting coefficient
    # np.savetxt("scaling.txt", list(zip(scales, Ns)))
    #m = sgd_randomized_pipe.best_estimator_.named_steps['clf'].coef_
    m = lr_model.named_steps['clf'].coef_
    print ("TEST HD", m)
    return -m[0]  # -lr_model.coef_#global_configuration.lr_model.coef_

#if __name__=='__main__':
	#cost_function(points = image)
