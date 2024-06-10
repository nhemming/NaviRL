"""
Graphs for reward analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import minimize

def derivFunc(g,n):
    r = 1.0
    return np.power(g, float(n)) * r * np.log(g)
    #return r*(1.0-g*g)*np.power(g,float(n))

def graphMaxAtHorizon():
    sns.set_theme()
    x = np.arange(1, 51)
    gamma = []
    direct_ag = []

    for n in x:
        sol = minimize(derivFunc, x0=np.array([0.5]), args=(n,), method='Nelder-Mead', bounds=((1e-10,1.0),))
        gamma.append(sol.x[0])

        #direct_ag.append(np.power())

    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,gamma)
    ax.set_xlabel('Number of Steps from Reward [-]')
    ax.set_ylabel('Maximizing Discount Factor [-]')
    ax.set_title('Discount Factor Maximizing Action Gap')

    print('Horizon\tGamma')
    for n,g in zip(x,gamma):
        print('{:d}\t{:.5f}'.format(n,g))

def graphDeriv():
    sns.set_theme()
    x = np.arange(1,101)
    gamma = [0.9,0.95,0.99]
    r = 1.0

    fig = plt.figure(0)
    ax = fig.add_subplot(1,1,1)

    for g in gamma:
        deriv = []
        for n in x:
            deriv.append(np.power(g,float(n))* r*np.log(g))

        ax.plot(x,np.abs(deriv),label=str(g))

    ax.set_xlabel('Number of Steps from Reward [-]')
    ax.set_ylabel('Absolute Value of Derivative of Reward [-]')
    ax.set_title('Derivative of Reward Felt at N Steps Away From\nReward State with Different Discount Factors')
    plt.tight_layout()
    ax.legend()

if __name__ == '__main__':

    graphDeriv()

    graphMaxAtHorizon()

    plt.show()