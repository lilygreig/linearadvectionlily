#explicit scheme uses the answer to work out the answer!
# we need to solve the set of simultaneous equations using a matrix 
# the needed matrix will be of size nxn where n is the number of x steps, or points between xmin and xmaimport 

import numpy as np
from numpy.linalg import solve

def BTCS(phiOld, c, nt):
    
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
    
    # in main text need from numpy.linalg import inv
    
    phi=phiOld.copy()
    for it in range(nt):
        phi=np.linalg.solve(M,phi)
       
        
    return phi
    
     