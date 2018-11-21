# Numerical schemes for simulating linear advection for outer code
# linearAdvect.py 

# The numpy package for numerical functions and pi
import numpy as np
from numpy.linalg import solve

def FTCS(phiOld, c, nt):
    "Linear advection of profile in phiOld using FTCS, Courant number c"
    "for nt time-steps"
    
    nx = len(phiOld)

    # new time-step array for phi
    phi = phiOld.copy()

    # FTCS for each time-step
    for it in range(nt):
        # Loop through all space using remainder after division (%)
        # to cope with periodic boundary conditions
        for j in range(nx):
            phi[j] = phiOld[j] - 0.5*c*\
                     (phiOld[(j+1)%nx] - phiOld[(j-1)%nx])
        
        # update arrays for next time-step
        phiOld = phi.copy()

    return phi

def FTBS(phiOld, c, nt):
    "Linear advection of profile in phiOld using FTBS, Courant number c"
    "for nt time-steps"
    
    nx = len(phiOld)

    # new time-step array for phi
    phi = phiOld.copy()

    # FTBS for each time-step
    for it in range(nt):
        # Loop through all space using remainder after division (%)
        # to cope with periodic boundary conditions
        for j in range(nx):
            phi[j] = phiOld[j] - c*\
                     (phiOld[j] - phiOld[(j-1)%nx])
        
        # update arrays for next time-step
        phiOld = phi.copy()

    return phi


def CTCS(phiOld, c, nt):
    "Linear advection of profile in phiOld using CTCS, Courant number c"
    "for nt time-steps"
    #explicit scheme computing future values of phi using values from the current time step and from the previous time step
    #number of elements in phiOld
    nx = len(phiOld)

    # new time-step array for phi
    phi = phiOld.copy()
    phivold= phiOld.copy()
    phi2=phiOld.copy()
    #use the scheme FTCS for one time step
    phi2 = FTCS(phiOld, c, 1)
      
    # CTCS for each time-step
    for it in range(nt-1):
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

def BTCS(phiOld, c, nt):
    "Linear advection of profile in phiOld using BTCS, Courant number c"
    "for nt time-steps"
    #the number of elements contained in phiOld
    nx = len(phiOld)
    M=np.zeros((nx,nx))
    # set diagonal values to be 1-c
    for i in range(nx):
        M[i,i]=(1)
    #set above and below diagonals to be c/2 
    for j in range(nx-1):
        M[j,j+1]=c/2
        M[j+1,j]=(-c/2)
        
    #finally for the boundaries
    M[nx-1,0]=c/2
    M[0,nx-1]=(-c/2)
    
    #need solve from numpy.linalg
    
    for it in range(nt):
        phiOld=np.linalg.solve(M,phiOld)
       
        
    return phiOld

def lax(phiOld, c, nt):
    "Linear advection of profile in phiOld using lax wendroff finite volume method, Courant number c"
    "for nt time-steps"
    
    nx = len(phiOld)

    # new time-step array for  phi
    phi = phiOld.copy()
    # lax for each time-step
    for it in range(nt):
        # Loop through all space using remainder after division (%)
        # to cope with periodic boundary conditions
        phiminus=0.5*(1+c)*phiOld[nx-1] +0.5*(1-c)*phiOld[0]
        for j in range(nx):
            #lax wendroff 
            phiplus = 0.5*(1+c)*phiOld[j] +0.5*(1-c)*phiOld[(j+1)%nx]
            #finite volume
            phi[j]=phiOld[j]-c*(phiplus-phiminus)
            #update phiminus
            phiminus=phiplus.copy() 
        # update arrays for next time-step
        phiOld = phi.copy()
         
    return phi





