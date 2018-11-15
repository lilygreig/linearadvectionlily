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
    TVCTCS_tot=np.zeros(nt-2)
    TVlax=np.zeros(nx)
    TVFTBS=np.zeros(nx)
    TVlax_tot=np.zeros(nt-2)
    TVFTBS_tot=np.zeros(nt-2)
    for i in range(2,nt):
        phiCTCS2 = CTCS(phiOld.copy(), c, i)
        philax2 = lax(phiOld.copy(), c, i)
        phiFTBS2 = FTBS(phiOld.copy(), c, i)
        for j in range(nx):
            TVCTCS[j]=abs(phiCTCS2[(j+1)%nx]-phiCTCS2[j])
            TVlax[j]=abs(philax2[(j+1)%nx]-philax2[j])
            TVFTBS[j]=abs(phiFTBS2[(j+1)%nx]-phiFTBS2[j])
        TVCTCS_tot[i-2]=sum(TVCTCS)
        TVlax_tot[i-2]=sum(TVlax)
        TVFTBS_tot[i-2]=sum(TVFTBS)
    return TVCTCS_tot, TVFTBS_tot, TVlax_tot


def order(xmin, xmax, nx_min, nx_max, c):
    "calculates order of convergence"
    dnx= nx_max-nx_min
    #nt = 5
    dx2=np.zeros(dnx)
    phiCTCSex=np.zeros(dnx)
    philaxex=np.zeros(dnx)
    phiBTCSex=np.zeros(dnx)
    phiFTBSex=np.zeros(dnx)
    phiFTCSex=np.zeros(dnx)
    for i in range(nx_min,nx_max):
        nx2 = i 
        dx2[i-nx_min] = (xmax - xmin)/nx2 
        nt2 = nx2
        x = np.arange(xmin, xmax, dx2[i-nx_min]) 
        phian=cosBell((x - c*nt2*dx2[i-nx_min])%(xmax - xmin), 0, 0.75)
        phiOld2=cosBell(x, 0, 0.75)
        philax2=lax(phiOld2.copy(), c, nt2)
        phiFTBS2=FTBS(phiOld2.copy(), c, nt2)
        phiCTCS2=CTCS(phiOld2.copy(), c, nt2)
        phiBTCS2=BTCS(phiOld2.copy(), c, nt2)
        phiFTCS2=FTCS(phiOld2.copy(), c, nt2)
        philaxex[i-nx_min]=l2ErrorNorm(philax2, phian)
        phiFTBSex[i-nx_min]=l2ErrorNorm(phiFTBS2, phian)
        phiCTCSex[i-nx_min]=l2ErrorNorm(phiCTCS2, phian)
        phiBTCSex[i-nx_min]=l2ErrorNorm(phiBTCS2, phian)
        phiFTCSex[i-nx_min]=l2ErrorNorm(phiFTCS2, phian)
    # do something with these slopes!! atm they are not outputted
    laxslope, laxintercept = np.polyfit(np.log(dx2), np.log(philaxex), 1)
    #print("lax slope =", laxslope)
    ftcsslope, ftcsintercept = np.polyfit(np.log(dx2), np.log(phiFTCSex), 1)
    #print("ftcs slope =",ftcsslope)
    ctcsslope, ctcsintercept = np.polyfit(np.log(dx2), np.log(phiCTCSex), 1)
    #print("ctcs slope =", ctcsslope)
    btcsslope, btcsintercept = np.polyfit(np.log(dx2), np.log(phiBTCSex), 1)
    #print("btcs slope =", btcsslope)
    ftbsslope, ftbsintercept = np.polyfit(np.log(dx2), np.log(phiFTBSex), 1)
    #print("ftbs slope =", ftbsslope)
    dx3=(dx2*dx2)
    # set dx slope so that the first value is the first value of ftbs
    #but these value will be logged
    dx21=min(philaxex)*dx2/min(dx2)
    dx32=min(philaxex)*dx3/min(dx3)
    phiFTBSex=phiFTBSex*min(philaxex)/min(phiFTBSex)
    phiCTCSex=phiCTCSex*min(philaxex)/min(phiCTCSex)
    phiBTCSex=phiBTCSex*min(philaxex)/min(phiBTCSex)
    philaxex=philaxex*min(philaxex)/min(philaxex)
    phiFTCSex=phiFTCSex*min(philaxex)/min(phiFTCSex)
    
    return phiFTBSex, phiCTCSex, phiFTCSex, phiBTCSex, philaxex, dx2, dx21, dx32

