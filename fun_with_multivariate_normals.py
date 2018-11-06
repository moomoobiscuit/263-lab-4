# ENGSCI263: Tutorial Lab 4 - Lumped Parameter Models
# fun_with_multivariate_normals.py

# PURPOSE:
# To INTRODUCE you to multivariate normal distributions.

# PREPARATION:
# TASKS 1 and 2 in main.py

# INSTRUCTIONS:
# turn on/off the code blocks below by toggling True/False 

# import modules and functions
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.linalg import det, inv

# Part I: 
    # Plot a normal distribution
    # - does it look like a normal distribution?
if False:
    # parameters
    mu,std = [1,2]                                          # distribution parameters
    th = np.linspace(mu-2*std,mu+2*std,41)                  # variable range for plotting
    P = np.exp(-(th-mu)**2/std**2)                          # prob. dist.
    # plotting commands
    f,ax=plt.subplots(1,1)
    ax.plot(th,P,'k-')
    ax.set_xlabel(r'$\theta_1$');ax.set_ylabel(r'$P$')
    plt.show()

# Part II:
    # Plot a 2D multivariate normal distribution
    # - does the mesh plot look like "a normal distribution but in 2d"?
    # - align the plot to look down the th1-axis - what does it look like?
    # - align the plot to look down the th2-axis - what does it look like?
    # - in which direction is the distribution the narrowest?
    # - align the plot to look down the P-axis - what does it look like?
    # - are the parameters th1 and th2 correlated?
if False:
    # parameters
    mu1,std1 = [1,2]                                        # distribution parameters
    mu2,std2 = [2,4]            
        # variable range for plotting
    th = np.linspace(np.min([mu1-2*std1,mu2-2*std2]),np.max([mu1+2*std1,mu2+2*std2]),41)
    TH1, TH2 = np.meshgrid(th,th,indexing='xy')             # plotting grid
        # prob. dist.
    P = np.exp(-0.5*((TH1-mu1)**2/std1**2+(TH2-mu2)**2/std2**2))/(2*np.pi*std1*std2)
    # plotting commands
    f,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    ax.plot_surface(TH1, TH2, P, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5,edgecolor='k')	
    ax.set_xlabel(r'$\theta_1$');ax.set_ylabel(r'$\theta_2$');ax.set_zlabel(r'$P$')
    plt.show()
    
# Part III:
    # Plot a 2D multivariate normal distribution with CORRELATED parameters
    # - set rho = 0 - how does the plot change?
    # - set rho = 0.7 - how does the plot change?
    # - set rho = 0.99 - how does the plot change?
    # - set rho = 1.01 - what happens?
if True:
    # parameters
    mu1,std1 = [1,2]                                        # distribution parameters
    mu2,std2 = [2,4] 
    rho = -0.7                                               # correlation parameter
        # variable range for plotting
    th = np.linspace(np.min([mu1-2*std1,mu2-2*std2]),np.max([mu1+2*std1,mu2+2*std2]),41)
    TH1, TH2 = np.meshgrid(th,th)                           # plotting grid
        # prob. dist.
    P = np.exp(-0.5/(1-rho**2)*((TH1-mu1)**2/std1**2+(TH2-mu2)**2/std2**2-
        2*rho*(TH1-mu1)*(TH2-mu2)/(std1*std2)))/(2*np.pi*std1*std2*np.sqrt(1-rho**2))
    # plotting commands
    f,ax=plt.subplots(1,1,subplot_kw={'projection':'3d'})
    ax.plot_surface(TH1, TH2, P, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5,edgecolor='k')	
    ax.set_xlabel(r'$\theta_1$');ax.set_ylabel(r'$\theta_2$');ax.set_zlabel(r'$P$')
    plt.show()