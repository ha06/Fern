import math
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class global_configuration():
	nvar = 12
	beta = 8
	iterations = 600
	varsize = [0, nvar]
	varmin = -.4
	varmax = 0.4
	varmin_param_1_2 = -0.1
	varmax_param_1_2 = 0.1
	varmin_param_3_4 = 0.80
	varmax_param_3_4 = 0.95
	maxit = 2#30
	npop = 10
	pc = 0.8
	nc = 2*math.ceil(pc*npop/2)  # Number of Offsprings (Parnets)
	pm = 0.4 # Mutation Percentage
	nm = math.ceil(pm*npop)      # Number of Mutants
	gamma = 0.02
	mu = 0.02
	#lr_model = SGDRegressor(max_iter=600, tol=1e-3)
	#lr =make_pipeline(StandardScaler(), lr_model)

	lr_model = SGDRegressor( max_iter=1000, tol=1e-3, eta0=0.1, penalty="elasticnet", fit_intercept=True)#,loss='huber')
	lr = make_pipeline(StandardScaler(), lr_model)
	#SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
	def init(self):
		return
