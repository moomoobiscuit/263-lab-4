
#########################################################################################
#
# Function library for plotting lumped parameter model
#
# 	Functions:
#		lmp_2D: lumped parameter model with 2 variables
#		lmp_3D: lumped parameter model with 3 variables
#		obj: objective function for a lmp
#
#########################################################################################

# import modules and functions
import numpy as np

# global variables - observations
tq,q = np.genfromtxt('wk_production_history.csv', delimiter = ',').T
tp,p = np.genfromtxt('wk_pressure_history.csv', delimiter = ',').T
dqdt = 0.*q                 # allocate derivative vector
dqdt[1:-1] = (q[2:]-q[:-2])/(tq[2:]-tq[:-2])    # central differences
dqdt[0] = (q[1]-q[0])/(tq[1]-tq[0])             # forward difference
dqdt[-1] = (q[-1]-q[-2])/(tq[-1]-tq[-2])        # backward difference

# define derivative function
def lpm(pi,t,a,b,c):                 # order of variables important
	''' ODE for lumped parameter model

		Parameters:
		-----------
		pi : float
			Pressure change from initial.
		t : float
			Time.
		a : float
			Parameter controlling drawdown response.
		b : float
			Parameter controlling recharge response.
		c : float
			Parameter controlling slow drainage.

		Returns:
		--------
		dpdt : float
			Rate of change of pressure in the reservoir.

	'''
	qi = np.interp(t,tq,q)           # interpolate (piecewise linear) flow rate
	dqdti = np.interp(t,tq,dqdt)     # interpolate derivative
	return -a*qi - b*pi - c*dqdti    # compute derivative

# implement an improved Euler step to solve the ODE
def solve_lpm(t,a,b,c=0):
	''' Solve the lumped parameter ODE.

		Parameters:
		-----------
		t : array-like
			Times at which to output solution.
		a : float
			Parameter controlling drawdown response.
		b : float
			Parameter controlling recharge response.
		c : float
			Parameter controlling slow drainage.

		Returns:
		--------
		p : array-like
			Pressure at times t.
	'''
	pm = [p[0],]                            # initial value
	for t0,t1 in zip(tp[:-1],tp[1:]):           # solve at pressure steps
		dpdt1 = lpm(pm[-1]-p[0], t0, a, b, c)   # predictor gradient
		pp = pm[-1] + dpdt1*(t1-t0)             # predictor step
		dpdt2 = lpm(pp-p[0], t1, a, b, c)       # corrector gradient
		pm.append(pm[-1] + 0.5*(t1-t0)*(dpdt2+dpdt1))  # corrector step
	return np.interp(t, tp, pm)             # interp onto requested times

def fit_mvn(parspace, dist):
	"""Finds the parameters of a multivariate normal distribution that best fits the data

    Parameters:
	-----------
		parspace : array-like
			list of meshgrid arrays spanning parameter space
		dist : array-like 
			PDF over parameter space
	Returns:
	--------
		mean : array-like
			distribution mean
		cov : array-like
			covariance matrix		
    """
	
	# dimensionality of parameter space
	N = len(parspace)
	
	# flatten arrays
	parspace = [p.flatten() for p in parspace]
	dist = dist.flatten()
	
	# compute means
	mean = [np.sum(dist*par)/np.sum(dist) for par in parspace]
	
	# compute covariance matrix
		# empty matrix
	cov = np.zeros((N,N))
		# loop over rows
	for i in range(0,N):
			# loop over upper triangle
		for j in range(i,N):
				# compute covariance
			cov[i,j] = np.sum(dist*(parspace[i] - mean[i])*(parspace[j] - mean[j]))/np.sum(dist)
				# assign to lower triangle
			if i != j: cov[j,i] = cov[i,j]
			
	return np.array(mean), np.array(cov)
		