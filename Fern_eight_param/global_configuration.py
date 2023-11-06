import math
from sklearn.linear_model import SGDRegressor
#from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class global_configuration():
	nvar = 8
	beta = 8
	iterations = 40000
	varsize = [0, nvar]
	varmin = -0.4
	varmax = 0.4
	maxit = 40#30
	npop = 100
	#nmax = 12
	pc = 0.7
	#nc = 2*math.ceil(pc*nmax/2)  # Number of Offsprings (Parnets)
	nc = 2 * math.ceil(pc * npop / 2)
	pm = 0.2# Mutation Percentage
	nmax = 20 #maximum number of population in any iteration
	nm = math.ceil(pm*npop)      # Number of Mutants
	#nm = math.ceil(pm * nmax)
	gamma = 0.06
	mu = 0.02
	#lr_model = SGDRegressor(max_iter=600, tol=1e-3)
	#lr =make_pipeline(StandardScaler(), lr_model)

	lr_model = SGDRegressor( max_iter=1000, tol=1e-3, eta0=0.1, penalty="elasticnet", fit_intercept=True)#,loss='huber')
	lr = make_pipeline(StandardScaler(), lr_model)
	#SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
	def init(self):
		return
