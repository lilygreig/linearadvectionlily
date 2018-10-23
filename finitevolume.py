# Numerical schemes for simulating linear advection for outer code
# linearAdvect.py 

# If you are using Python 2.7 rather than Python 3, import various
# functions from Python 3 such as to use real number division
# rather than integer division. ie 3/2  = 1.5  rather than 3/2 = 1
#from __future__ import absolute_import, division, print_function

# The numpy package for numerical functions and pi
import numpy as np

def lax(phiOld, c, nt):
    "Linear advection of profile in phiOld using FTCS, Courant number c"
    "for nt time-steps"
    
    nx = len(phiOld)

    # new time-step array for  phi
    phi = phiOld.copy()
    phiplus=phiOld.copy()
    phiminus=phiOld.copy()
    # FTCS for each time-step
    for it in range(nt):
        # Loop through all space using remainder after division (%)
        # to cope with periodic boundary conditions
        
        for j in range(nx):
            phiplus[j] = 0.5*(1+c)*phiOld[j] +0.5*(1-c)*phiOld[(j+1)%nx]
            phiminus[j]= 0.5*(1+c)*phiOld[(j-1)%nx] +0.5*(1-c)*phiOld[j]
            phi[j]=phiOld[j]-c*(phiplus[j]-phiminus[j])
              
        # update arrays for next time-step
        phiOld = phi.copy()
         
    return phi