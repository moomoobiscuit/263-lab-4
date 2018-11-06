
#########################################################################################
#
# Function library for plotting lumped parameter results
#
# 	Functions:
#		plot_lmp: plot one instance of the lmp model compared to the observation
#		plot_posterior: surface plot of the posterior
#		plot_samples: plot samples on top of the posterior
#
#########################################################################################

# import modules and functions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from lumped_parameter_model import *

# set font size
text_size = 16.

def plot_lmp(lmp, theta):
	"""Plot lmp model

    Args:
		lmp (callable): lumped parameter model
		theta (numpy array): lmp parameters vector
    """
	
	# write label for model
	a = str(round(theta[0]*1.e-8*365*24*3600, 2))
	b = str(round(theta[1], 2))
	label = 'Model for: \n'
	label += r'$a='+a+'years^{-1}$'
	label += '\n'+r'$b='+b+'10^{-5}m^{-1}s^{-2}$'
	if len(theta) == 3:	# check model dimension 
		c = str(round(theta[2], 2))
		label += '\n'+r'$c='+c+'10^{3}m^{-1}s^{-1}$'
		
	# plot parameters
	year_end_calib = 1976.	# interval used for calibration: observations till this date
	
	# load pressure history
	table = np.loadtxt(open('pressure_history.csv','rb'), delimiter = ',')
	t = table[:,0]		# time vector [years]
	p_real = table[:,1]	# observed reservoir pressure vector [bars]
	imax = np.argmin(abs(t-year_end_calib))	# index of the final year used in calibration
	
	# load model
	p_model = lmp(theta)	# simulated reservoir pressure vector [bars]
	
	# plotting
	fig = plt.figure(figsize = [10., 7.])			# open figure
	ax1 = plt.axes()								# create axes
	ax1.plot(t[:imax+1], p_real[:imax+1], 'bx', mew = 1.5, zorder = 10, lw = 1.5, label = 'Observations')	# show observations
	ax1.plot(t[:imax+1], p_model[:imax+1], color ='k', lw = 1.5, label = label)					# show model
	
	# plotting upkeep
	ax1.set_xlim(t[0], t[imax])
	ylim = [.4*min(p_real), 1.1*max(p_real)]
	ax1.set_ylim(ylim)
	ax1.set_xlabel('Time (years)', fontsize = text_size)
	ax1.set_ylabel('Pressure (bars)', fontsize = text_size)
	ax1.legend(loc = 'upper right', fontsize = text_size, framealpha = 0.)
	ax1.tick_params(labelsize=text_size)
	
	# save and show
	if len(theta) == 2: save_name = 'lab4_plot1.png'
	elif len(theta) == 3: save_name = 'lab4_plot4.png'
	plt.savefig(save_name, bbox_inches = 'tight')
	plt.show()
	
def plot_posterior(a,b,c=None,P=None):
	if c is None:
		plot_posterior2D(a,b,P)
	else:
		plot_posterior3D(a,b,c,P)

def plot_posterior2D(a, b, P):	
	"""Plot posterior distribution

    Args:
		a (numpy array): a distribution vector
		b (numpy array): b distribution vector
		P (numpy array): posterior matrix
    """
	
	# grid of parameter values: returns every possible combination of parameters in a and b
	A, B = np.meshgrid(a, b)
	
	# plotting
	fig = plt.figure(figsize=[10., 7.])				# open figure
	ax1 = fig.add_subplot(111, projection='3d')		# create 3D axes
	ax1.plot_surface(A, B, P, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5,edgecolor='k')	# show surface
	
	# plotting upkeep
	ax1.set_xlabel('a', fontsize = text_size)
	ax1.set_ylabel('b', fontsize = text_size)
	ax1.set_zlabel('P', fontsize = text_size)
	ax1.set_xlim([a[0], a[-1]])
	ax1.set_ylim([b[0], b[-1]])
	ax1.set_zlim(0., )
	ax1.view_init(40, 100.)
	
	# save and show
	plt.show()
	
def plot_posterior3D(a, b, c, P):	
	"""Plot posterior distribution for each parameter combination

    Args:
		a (numpy array): a distribution vector
		b (numpy array): b distribution vector
		P (numpy array): posterior matrix
    """
	
	# plotting variables
	azim = 15.		# azimuth at which surfaces are shown
	
	# a and b combination
	Ab, Ba = np.meshgrid(a, b, indexing='ij')
	Pab = np.zeros(Ab.shape)
	for i in range(len(a)):
		for j in range(len(b)):
			Pab[i][j] = sum([P[i][j][k] for k in range(len(c))])

	# a and c combination			
	Ac, Ca = np.meshgrid(a, c, indexing='ij')
	Pac = np.zeros(Ac.shape)
	for i in range(len(a)):
		for k in range(len(c)):
			Pac[i][k] = sum([P[i][j][k] for j in range(len(b))])
	
	# b and c combination		
	Bc, Cb = np.meshgrid(b, c, indexing='ij')
	Pbc = np.zeros(Bc.shape)
	for j in range(len(b)):
		for k in range(len(c)):
			Pbc[j][k] = sum([P[i][j][k] for i in range(len(a))])
			
	# plotting
	fig = plt.figure(figsize=[20.0,15.])
	ax1 = fig.add_subplot(221, projection='3d')
	ax1.plot_surface(Ab, Ba, Pab, rstride=1, cstride=1, cmap=cm.Oranges, lw = 0.5)
	ax1.set_xlabel('a')
	ax1.set_ylabel('b')
	ax1.set_zlabel('P')
	ax1.set_xlim([a[0], a[-1]])
	ax1.set_ylim([b[0], b[-1]])
	ax1.set_zlim(0., )
	ax1.view_init(40, azim)
	
	ax1 = fig.add_subplot(222, projection='3d')
	ax1.plot_surface(Ac, Ca, Pac, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)
	ax1.set_xlabel('a')
	ax1.set_ylabel('c')
	ax1.set_zlabel('P')
	ax1.set_xlim([a[0], a[-1]])
	ax1.set_ylim([c[0], c[-1]])
	ax1.set_zlim(0., )
	ax1.view_init(40, azim)

	ax1 = fig.add_subplot(223, projection='3d')
	ax1.plot_surface(Bc, Cb, Pbc, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)
	ax1.set_xlabel('b')
	ax1.set_ylabel('c')
	ax1.set_zlabel('P')
	ax1.set_xlim([b[0], b[-1]])
	ax1.set_ylim([c[0], c[-1]])
	ax1.set_zlim(0., )
	ax1.view_init(40, azim)
	
	# save and show
	plt.show()

def plot_samples(a, b, c=None, P=None, samples=None):
	if c is None:
		plot_samples2D(a,b,P,samples)
	else:
		plot_samples3D(a,b,c,P,samples)

def plot_samples2D(a, b, P, samples):
	# plotting
	fig = plt.figure(figsize=[10., 7.])				# open figure
	ax1 = fig.add_subplot(111, projection='3d')		# create 3D axes
	A, B = np.meshgrid(a, b, indexing='ij')
	ax1.plot_surface(A, B, P, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)	# show surface
	
	tp,po = np.genfromtxt('wk_pressure_history.csv', delimiter = ',')[:28,:].T
	v = 2
	s = np.array([np.sum((solve_lpm(tp,a,b)-po)**2)/v for a,b in samples])
	p = np.exp(-s/2.)
	p = p/np.max(p)*np.max(P)*1.2
		
	ax1.plot(*samples.T,p,'k.')

	# plotting upkeep
	ax1.set_xlabel('a', fontsize = text_size)
	ax1.set_ylabel('b', fontsize = text_size)
	ax1.set_zlabel('P', fontsize = text_size)
	ax1.set_zlim(0., )
	ax1.view_init(40, 100.)
	
	# save and show
	plt.show()

def plot_samples3D(a, b, c, P, samples):
	# plotting variables
	azim = 15.		# azimuth at which surfaces are shown
	
	# a and b combination
	Ab, Ba = np.meshgrid(a, b, indexing='ij')
	Pab = np.zeros(Ab.shape)
	for i in range(len(a)):
		for j in range(len(b)):
			Pab[i][j] = sum([P[i][j][k] for k in range(len(c))])

	# a and c combination			
	Ac, Ca = np.meshgrid(a, c, indexing='ij')
	Pac = np.zeros(Ac.shape)
	for i in range(len(a)):
		for k in range(len(c)):
			Pac[i][k] = sum([P[i][j][k] for j in range(len(b))])
	
	# b and c combination		
	Bc, Cb = np.meshgrid(b, c, indexing='ij')
	Pbc = np.zeros(Bc.shape)
	for j in range(len(b)):
		for k in range(len(c)):
			Pbc[j][k] = sum([P[i][j][k] for i in range(len(a))])

	tp,po = np.genfromtxt('wk_pressure_history.csv', delimiter = ',')[:28,:].T
	v = 2
	s = np.array([np.sum((solve_lpm(tp,a,b,c)-po)**2)/v for a,b,c in samples])
	p = np.exp(-s/2.)
	p = p/np.max(p)*np.max(P)*1.2

	# plotting
	fig = plt.figure(figsize=[20.0,15.])
	ax1 = fig.add_subplot(221, projection='3d')
	ax1.plot_surface(Ab, Ba, Pab, rstride=1, cstride=1, cmap=cm.Oranges, lw = 0.5)
	ax1.set_xlabel('a')
	ax1.set_ylabel('b')
	ax1.set_zlabel('P')
	ax1.set_xlim([a[0], a[-1]])
	ax1.set_ylim([b[0], b[-1]])
	ax1.set_zlim(0., )
	ax1.view_init(40, azim)
	ax1.plot(samples[:,0],samples[:,1],p,'k.')
	
	ax1 = fig.add_subplot(222, projection='3d')
	ax1.plot_surface(Ac, Ca, Pac, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)
	ax1.set_xlabel('a')
	ax1.set_ylabel('c')
	ax1.set_zlabel('P')
	ax1.set_xlim([a[0], a[-1]])
	ax1.set_ylim([c[0], c[-1]])
	ax1.set_zlim(0., )
	ax1.view_init(40, azim)
	ax1.plot(samples[:,0],samples[:,-1],p,'k.')

	ax1 = fig.add_subplot(223, projection='3d')
	ax1.plot_surface(Bc, Cb, Pbc, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)
	ax1.set_xlabel('b')
	ax1.set_ylabel('c')
	ax1.set_zlabel('P')
	ax1.set_xlim([b[0], b[-1]])
	ax1.set_ylim([c[0], c[-1]])
	ax1.set_zlim(0., )
	ax1.view_init(40, azim)
	ax1.plot(samples[:,1],samples[:,-1],p,'k.')
	
	# save and show
	plt.show()
