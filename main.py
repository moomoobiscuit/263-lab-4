# ENGSCI263: Tutorial Lab 4 - Lumped Parameter Models
# main.py

# PURPOSE:
# To COMPUTE a posterior distribution and use it to CONSTRUCT an ensemble of models.
# To EXAMINE structural error.

# PREPARATION:
# Notebook uncertainty.ipynb

# SUBMISSION:
# There is NOTHING to submit for this lab.

# INSTRUCTIONS:
# Jump to section "if __name__ == "__main__":" at bottom of this file.

# import modules and functions
import numpy as np
from lumped_parameter_model import *
from plotting import *

####################################################################################
#
# Task 1: Ad hoc calibration of the lumped parameter model.
#
####################################################################################
def get_familiar_with_model():
	''' This function runs and plots the lumped parameter model for your selection of 
		model parameters.
	'''
	# **to do**
	# 1. CHOOSE values of a and b that provide a good fit to the model
	# *stuck?* look at Section 3.4.2 in the Uncertainty notebook
	# 2. CALCULATE the sum of squares objective function
	# 3. ANSWER the questions in the lab document

	# set model parameters (we'll leave c=0 for now)
	a = .0047 
	b = .26
	
	# get data and run model
		# po = pressure observation
	tp,po = np.genfromtxt('wk_pressure_history.csv', delimiter = ',')[:28,:].T
		# pm = pressure model
	pm = solve_lpm(tp,a,b,c=0)
	
	# error variance - 2 bar
	v = 2.

	# 2. calculate the sum-of-squares objective function (= 0. just a placeholder)
	S = np.sum((pm-po)**2) / v
	
	# plotting commands
	f,ax = plt.subplots(1,1)
	ax.plot(tp,pm,'b-', label='model')
	ax.errorbar(tp,po,yerr=v,fmt='ro', label='data')
	ax.set_xlabel('time')
	ax.set_ylabel('pressure')
	ax.set_title('objective function: S={:3.2f}'.format(S))
	ax.legend()
	plt.show()

####################################################################################
#
# Task 2: Grid search to construct posterior.
#
####################################################################################
def grid_search():
	''' This function implements a grid search to compute the posterior over a and b.

		Returns:
		--------
		a : array-like
			Vector of 'a' parameter values.
		b : array-like
			Vector of 'b' parameter values.
		P : array-like
			Posterior probability distribution.
	'''
	# **to do**
	# 1. DEFINE parameter ranges for the grid search
	# 2. COMPUTE the sum-of-squares objective function for each parameter combination
	# 3. COMPUTE the posterior probability distribution
	# 4. ANSWER the questions in the lab document

	# 1. define parameter ranges for the grid search
	a_best = .0047
	b_best = .26

	# number of values considered for each parameter within a given interval
	N = 51	

	# vectors of parameter values
	a = np.linspace(a_best/2,a_best*1.5, N)
	b = np.linspace(b_best/2,b_best*1.5, N)

	# grid of parameter values: returns every possible combination of parameters in a and b
	A, B = np.meshgrid(a, b, indexing='ij')

	# empty 2D matrix for objective function
	S = np.zeros(A.shape)

	# data for calibration
	tp,po = np.genfromtxt('wk_pressure_history.csv', delimiter = ',')[:28,:].T

	# error variance - 2 bar
	v = 1.

	# grid search algorithm
	for i in range(len(a)):
		for j in range(len(b)):
			# 2. compute the sum of squares objective function at each value 
			pm = solve_lpm(tp,a[i],b[j],c=0)
			S[i,j] = np.sum((pm-po)**2) / v

	# 3. compute the posterior
	P = np.exp(-S/2)
	
	# normalize to a probability density function
	Pint = np.sum(P)*(a[1]-a[0])*(b[1]-b[0])
	P = P/Pint

	# plot posterior parameter distribution
	#plot_posterior(a, b, P=P)

	return a,b,P
	
####################################################################################
#
# Task3: Open fun_with_multivariate_normals.py and complete the exercises.
#
####################################################################################
	
####################################################################################
#
# Task 4: Sample from the posterior.
#
####################################################################################
def construct_samples(a,b,P,N_samples):
	''' This function constructs samples from a multivariate normal distribution
	    fitted to the data.

		Parameters:
		-----------
		a : array-like
			Vector of 'a' parameter values.
		b : array-like
			Vector of 'b' parameter values.
		P : array-like
			Posterior probability distribution.
		N_samples : int
			Number of samples to take.

		Returns:
		--------
		samples : array-like
			parameter samples from the multivariate normal
	'''
	# **to do**
	# 1. FIGURE OUT how to use the multivariate normal functionality in numpy
	#    to generate parameter samples
	# 2. ANSWER the questions in the lab document

	# compute properties (fitting) of multivariate normal distribution
	# mean = a vector of parameter means
	# covariance = a matrix of parameter variances and correlations
	A, B = np.meshgrid(a,b,indexing='ij')
	mean, covariance = fit_mvn([A,B], P)

	# 1. create samples using numpy function multivariate_normal (Google it)
	samples = np.random.multivariate_normal(mean, covariance, N_samples)
	
	# plot samples and predictions
	#plot_samples(a, b, P=P, samples=samples)

	return samples
	
####################################################################################
#
# Task 5: Make predictions for your samples.
#
####################################################################################
def model_ensemble(samples):
	''' Runs the model for given parameter samples and plots the results.

		Parameters:
		-----------
		samples : array-like
			parameter samples from the multivariate normal
	'''
	# **to do**
	# Run your parameter samples through the model and plot the predictions.

	# 1. choose a time vector to evaluate your model between 1953 and 2012 
	t = np.linspace(1953,2012,50)
	
	# 2. create a figure and axes (see TASK 1)
	f,ax = plt.subplots(1,1)
	
	# 3. for each sample, solve and plot the model  (see TASK 1)
	for a,b in samples:
		pm = solve_lpm(t,a,b,c=0)
		ax.plot(t,pm,'b-', label='model')
		
		#*hint* use lw= and alpha= to set line width and transparency
		#pass  # this command just a placeholder, delete it when you write commands above
	
    # this command just adds a line to the legend
	ax.plot([],[],'k-', lw=0.2,alpha=0.2, label='model ensemble')

	# get the data
	tp,po = np.genfromtxt('wk_pressure_history.csv', delimiter = ',').T	
	ax.axvline(1980, color='b', linestyle=':', label='calibration/forecast')
	
	# 4. plot Wairakei data as error bars
	# *hint* see TASK 1 for appropriate plotting commands
	
	ax.errorbar(tp,po,yerr=2,fmt='ro', label='data')
	ax.set_xlabel('time')
	ax.set_ylabel('pressure')
	#ax.set_title('objective function: S={:3.2f}'.format(S))
	#ax.legend()
	plt.show()

####################################################################################
#
# Task 6: Do it all again in 3 dimensions (a, b and c).
#
####################################################################################


if __name__=="__main__":
	# Comment/uncomment each of the functions below as you complete the tasks
	
	# TASK 1: Read the instructions in the function definition.
	#get_familiar_with_model()
	
	# TASK 2: Read the instructions in the function definition.
	a,b,posterior = grid_search()
	
	# TASK 3: Open the file fun_with_multivariate_normals.py and complete the tasks.

	# TASK 4: Read the instructions in the function definition.
	# this task relies on the output of TASK 2, so don't comment that command
	N = 100
	samples = construct_samples(a, b, posterior, N)

	# TASK 5: Read the instructions in the function definition.
	# this task relies on the output of TASKS 2 and 3, so don't comment those commands
	model_ensemble(samples)

	# TASK 6: Copy main.py to the new file main_3D.py.
	# make changes to run the analysis again with c as a free parameter
	# plotting functions should respond to changes in inputs, i.e.,
	# - plot_posterior(a, b, P=P)  changes to  plot_posterior(a, b, c, P=P)
	#   providing you have sensibly defined c
	# - same for plot_samples(A, B, P=P, samples=samples)
	# fit_mvn() is set up to accept 2 or higher dimension posteriors
	# best fit parameters for a,b,c (from curve_fit) are [2.2e-3,1.1e-1,6.8e-3]
	# reduce N in grid_search or this will take forever
