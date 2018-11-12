#!/usr/bin/python3 

# will integrate this into main code once have got it working

import numpy as np
import matplotlib.pyplot as plt

# read in all the linear advection schemes, initial conditions and other
# code associated with this application
from initialConditions import * 
from advectionSchemes import *
from diagnostics import *
# The main code is inside a function to avoid global variables
def main():
    "Advect the initial conditions using various advection schemes and"
    "compare results"

    # Parameters
    xmin = 0
    xmax = 1
    nx_max = 60
    nx_min=40
    dnx= nx_max-nx_min
    #nt = 40
    c = 0.2
    dx2=np.zeros(dnx)
    phiCTCSex=np.zeros(dnx)
    philaxex=np.zeros(dnx)
    #phiBTCSex=np.zeros(dnx)
    phiFTBSex=np.zeros(dnx)
    #phiFTCSex=np.zeros(dnx)
    for i in range(nx_min,nx_max):
        nx2 = i+2 
        dx2[i-40] = (xmax - xmin)/nx2 
        nt2 = nx2*nx_min 
        x = np.arange(xmin, xmax, dx2[i-40]) 
        phian=cosBell((x - c*nt2*dx2[i-40])%(xmax - xmin), 0, 0.75)
        phiOld2=cosBell(x, 0, 0.75)
        philax2=lax(phiOld2.copy(), c, nt2)
        phiFTBS2=FTBS(phiOld2.copy(), c, nt2)
        phiCTCS2=FTBS(phiOld2.copy(), c, nt2)
        philaxex[i-40]=l2ErrorNorm(philax2, phian)
        phiFTBSex[i-40]=l2ErrorNorm(phiFTBS2, phian)
        phiCTCSex[i-40]=l2ErrorNorm(phiCTCS2, phian)
    # spatial points for plotting and for defining initial conditions
    dx3=dx2*dx2
    plt.figure(1)
    plt.plot(dx2,phiCTCSex, label='CTCS,l2', color="green", marker="*")
    #plt.plot(dx,phiBTCSex, label='BTCS,l2', color="red")
    #plt.plot(dx,phiFTCSex, label='FTCS,l2', color="blue")
    plt.plot(dx2,phiFTBSex, label='FTBS,l2', color="yellow", marker="")
    plt.plot(dx2,philaxex, label='lax,l2', color="magenta", marker=".")
    plt.plot(dx2,dx2, label='dx', color="orange", linestyle="--")
    plt.plot(dx2,dx3, label='dx^2', color="brown", linestyle="--")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$\Delta x$')
    plt.ylabel('l2')
    plt.legend(bbox_to_anchor=(1.15 , 1.1))
    plt.savefig('plots/error2.pdf', bbox_inches="tight")
    input('press return to save file and continue')
    
    
    plt.show()

main()