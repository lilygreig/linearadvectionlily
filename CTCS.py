#this explicit scheme computes future values of phi using values from the current time step and from the previous time step
import numpy as np

def CTCS(phiOld, c, nt):
    "Linear advection of profile in phiOld using FTCS, Courant number c"
    "for nt time-steps"
    
    #the number of elements contained in phiOld
    nx = len(phiOld)

    # new time-step array for phi
    phi = phiOld.copy()
    phivold= phiOld.copy()
    phi2=phiOld.copy()
    #how do i get n-1 for the first time step
    #we use initial conditions for the phi(n-1) but then we skip to phi(n+1) so how do we generate the points for phi(n)?
    #use the scheme FTCS
    #phiat time step n
    
    for j in range(nx):
        phi2[j] = phiOld[j] - 0.5*c*\
                     (phiOld[(j+1)%nx] - phiOld[(j-1)%nx])
    # CTCS for each time-step
    for it in range(nt):
        # Loop through all space using remainder after division (%)
        # to cope with periodic boundary conditions
        for j in range(nx): #update every element of the old phi
            phi[j] = phivold[j] - c*\
                     (phi2[(j+1)%nx] - phi2[(j-1)%nx])
        
        # update arrays for next time-step
        #update phi very old to be phi old
        phivold=phi2.copy()
        #update phiold to be new phi
        phi2 = phi.copy()
        

    return phi