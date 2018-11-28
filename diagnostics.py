# Various function for plotting results and for calculating error measures


import numpy as np
from advectionSchemes import *
from initialConditions import *

def l2ErrorNorm(phi, phiExact):
    "Calculates the l2 error norm (RMS error) of phi in comparison to"
    "phiExact"

    # calculate the error and the RMS error norm
    phiError = phi - phiExact
    l2 = np.sqrt(sum(phiError**2)/sum(phiExact**2))

    return l2


def lInfErrorNorm(phi, phiExact):
    "Calculates the linf error norm (maximum normalised error) in comparison"
    "to phiExact"
    phiError = phi - phiExact
    return np.max(np.abs(phiError))/np.max(np.abs(phiExact))

def l2t(phiOld, nt, c, xmax, xmin, dx, x):
    "Calculates l2 at each time step and stores it"
    phiCTCSet=np.zeros(nt)
    philaxet=np.zeros(nt)
    phiBTCSet=np.zeros(nt)
    phiFTBSet=np.zeros(nt)
    phiFTCSet=np.zeros(nt)
    for i in range(2,nt):
        phiCTCSt = CTCS(phiOld.copy(), c, i)
        phiBTCSt = BTCS(phiOld.copy(), c, i)
        phiFTCSt = FTCS(phiOld.copy(), c, i)
        phiFTBSt = FTBS(phiOld.copy(), c, i)
        philaxt = lax(phiOld.copy(), c, i)
        phiAnalytict = cosBell((x - c*i*dx)%(xmax - xmin), 0, 0.75)
        phiCTCSet[i]=l2ErrorNorm(phiCTCSt, phiAnalytict)
        philaxet[i]=l2ErrorNorm(philaxt, phiAnalytict)
        phiBTCSet[i]=l2ErrorNorm(phiBTCSt, phiAnalytict)
        phiFTBSet[i]=l2ErrorNorm(phiFTBSt, phiAnalytict)
        phiFTCSet[i]=l2ErrorNorm(phiFTCSt, phiAnalytict)
    return phiCTCSet, phiBTCSet, phiFTCSet, phiFTBSet, philaxet

def TV(phiOld, nt, c):
    "Calculates the total variation of a scheme with respect to time"
    nx=len(phiOld)
    TVCTCS=np.zeros(nx)
    TVlax=np.zeros(nx)
    TVFTBS=np.zeros(nx)
    TVFTCS=np.zeros(nx)
    TVBTCS=np.zeros(nx)
    # set min no. of time steps to 2
    TVCTCS_tot=np.zeros(nt-2)
    TVlax_tot=np.zeros(nt-2)
    TVFTBS_tot=np.zeros(nt-2)
    TVFTCS_tot=np.zeros(nt-2)
    TVBTCS_tot=np.zeros(nt-2)
    for i in range(2,nt):
        #advect profiles i timesteps
        phiCTCS2 = CTCS(phiOld.copy(), c, i)
        philax2 = lax(phiOld.copy(), c, i)
        phiFTBS2 = FTBS(phiOld.copy(), c, i)
        phiFTCS2 = FTCS(phiOld.copy(), c, i)
        phiBTCS2 = BTCS(phiOld.copy(), c, i)
        for j in range(nx):
            #variation for each spatial point
            TVCTCS[j]=abs(phiCTCS2[(j+1)%nx]-phiCTCS2[j])
            TVlax[j]=abs(philax2[(j+1)%nx]-philax2[j])
            TVFTBS[j]=abs(phiFTBS2[(j+1)%nx]-phiFTBS2[j])
            TVFTCS[j]=abs(phiFTCS2[(j+1)%nx]-phiFTCS2[j])
            TVBTCS[j]=abs(phiBTCS2[(j+1)%nx]-phiBTCS2[j])
        # sum to get total variation for i timesteps
        TVCTCS_tot[i-2]=sum(TVCTCS)
        TVlax_tot[i-2]=sum(TVlax)
        TVFTBS_tot[i-2]=sum(TVFTBS)
        TVFTCS_tot[i-2]=sum(TVFTCS)
        TVBTCS_tot[i-2]=sum(TVBTCS)
    return TVCTCS_tot, TVFTBS_tot, TVlax_tot, TVFTCS_tot, TVBTCS_tot


def order(xmin, xmax, nx_min, nx_max, c):
    "calculates order of convergence, calculates l2 error as dx changes"
    #range of nx
    dnx= nx_max-nx_min
    dx1=np.zeros(dnx)
    phiCTCSex=np.zeros(dnx)
    philaxex=np.zeros(dnx)
    phiBTCSex=np.zeros(dnx)
    phiFTBSex=np.zeros(dnx)
    phiFTCSex=np.zeros(dnx)
    for i in range(nx_min,nx_max):
        nx1 = i
        #calculate new dx and store it
        dx1[i-nx_min] = (xmax - xmin)/nx1
        # calculate new nt to keep c constant
        nt1 = nx1
        x = np.arange(xmin, xmax, dx1[i-nx_min])
        #analytic solution for these parameters
        phian=cosBell((x - c*nt1*dx1[i-nx_min])%(xmax - xmin), 0, 0.75)
        phiOld2=cosBell(x, 0, 0.75)
        #advect initial conditions using schemes and new parameters
        philax2=lax(phiOld2.copy(), c, nt1)
        phiFTBS2=FTBS(phiOld2.copy(), c, nt1)
        phiCTCS2=CTCS(phiOld2.copy(), c, nt1)
        phiBTCS2=BTCS(phiOld2.copy(), c, nt1)
        phiFTCS2=FTCS(phiOld2.copy(), c, nt1)
        # calculate l2 for new dx value
        philaxex[i-nx_min]=l2ErrorNorm(philax2, phian)
        phiFTBSex[i-nx_min]=l2ErrorNorm(phiFTBS2, phian)
        phiCTCSex[i-nx_min]=l2ErrorNorm(phiCTCS2, phian)
        phiBTCSex[i-nx_min]=l2ErrorNorm(phiBTCS2, phian)
        phiFTCSex[i-nx_min]=l2ErrorNorm(phiFTCS2, phian)
    dx2=(dx1*dx1)
    # set all slopes to plot from the same point
    # choose this point to be the minimum value of all the error
    minval=min(max(philaxex), max(phiFTBSex), max(phiCTCSex), max(phiBTCSex), max(phiFTCSex))
    #but all error values will be logged so instead of adding constant, multiply by
    #constants
    dx1=minval*dx1/min(dx1)
    dx2=minval*dx2/min(dx2)
    phiFTBSex=phiFTBSex*minval/min(phiFTBSex)
    phiCTCSex=phiCTCSex*minval/min(phiCTCSex)
    phiBTCSex=phiBTCSex*minval/min(phiBTCSex)
    philaxex=philaxex*minval/min(philaxex)
    phiFTCSex=phiFTCSex*minval/min(phiFTCSex)
    return phiFTBSex, phiCTCSex, phiFTCSex, phiBTCSex, philaxex, dx1, dx1, dx2


def stability(phiOld, dx, xmax, xmin, x, c, nt):
    "Plots l2 error for all schemes for different values of c, with"
    "varying nt to keep total time constant"
    # choose a constant for c*nt
    cst=nt*1.2
    errorCTCS=np.zeros(len(c))
    errorBTCS=np.zeros(len(c))
    errorFTCS=np.zeros(len(c))
    errorFTBS=np.zeros(len(c))
    errorlax=np.zeros(len(c))
    j=0
    for i in c:
        #keep c*nt as this constant to ensure total time remains the same
        nt=int(abs(cst/i))
        phiCTCSs=CTCS(phiOld.copy(), i, nt)
        phiBTCSs=BTCS(phiOld.copy(), i, nt)
        phiFTCSs=FTCS(phiOld.copy(), i, nt)
        phiFTBSs=FTBS(phiOld.copy(), i, nt)
        philaxs=lax(phiOld.copy(), i, nt)
        phian=cosBell((x - i*nt*dx)%(xmax - xmin), 0, 0.75)
        # calculate and store error for new c value for all schemes
        errorCTCS[j]=l2ErrorNorm(phiCTCSs, phian)
        errorBTCS[j]=l2ErrorNorm(phiBTCSs, phian)
        errorFTCS[j]=l2ErrorNorm(phiFTCSs, phian)
        errorFTBS[j]=l2ErrorNorm(phiFTBSs, phian)
        errorlax[j]=l2ErrorNorm(philaxs, phian)
        j=j+1
    return errorCTCS, errorBTCS, errorFTBS, errorFTCS, errorlax
