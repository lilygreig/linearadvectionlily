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
    nx = 40
    nt = 40
    c = 0.2
# ftbs is stable for c between 0 and 1

    # Derived parameters
    dx = (xmax - xmin)/nx

    # spatial points for plotting and for defining initial conditions
    x = np.arange(xmin, xmax, dx)

    # Initial conditions
    phiOld = cosBell(x, 0, 0.75)
    phiOldsw= squareWave(x, 0, 0.75)
    # Exact solution is the initial condition shifted around the domain
    phiAnalytic = cosBell((x - c*nt*dx)%(xmax - xmin), 0, 0.75)
    phiAnalyticsw = squareWave((x - c*nt*dx)%(xmax - xmin), 0, 0.75)
    # Advect the cosbell profile using finite difference for all the time steps
    phiFTCS = FTCS(phiOld.copy(), c, nt)
    phiFTBS = FTBS(phiOld.copy(), c, nt)
    phiCTCS = CTCS(phiOld.copy(), c, nt)
    phiBTCS = BTCS(phiOld.copy(), c, nt)
    philax= lax(phiOld.copy(), c, nt)
    # Advect the square wave profile
    phiFTCSsw = FTCS(phiOldsw.copy(), c, nt)
    phiFTBSsw = FTBS(phiOldsw.copy(), c, nt)
    phiCTCSsw = CTCS(phiOldsw.copy(), c, nt)
    phiBTCSsw = BTCS(phiOldsw.copy(), c, nt)
    philaxsw= lax(phiOldsw.copy(), c, nt)
    # Calculate and print out error norms for the cosbell initial conditions
    print("FTCS l2 error norm = ", l2ErrorNorm(phiFTCS, phiAnalytic))
    print("FTCS linf error norm = ", lInfErrorNorm(phiFTCS, phiAnalytic))
    print("FTBS l2 error norm = ", l2ErrorNorm(phiFTBS, phiAnalytic))
    print("FTBS linf error norm = ", lInfErrorNorm(phiFTBS, phiAnalytic))
    print("CTCS l2 error norm = ", l2ErrorNorm(phiCTCS, phiAnalytic))
    print("CTCS linf error norm = ", lInfErrorNorm(phiCTCS, phiAnalytic))
    print("BTCS l2 error norm = ", l2ErrorNorm(phiBTCS, phiAnalytic))
    print("BTCS linf error norm = ", lInfErrorNorm(phiBTCS, phiAnalytic))
    print("lax l2 error norm = ", l2ErrorNorm(philax, phiAnalytic))
    print("lax linf error norm = ", lInfErrorNorm(philax, phiAnalytic))
    # Calculate and print out error norms for the square wave initial conditions
    print("FTCS l2 error norm SW = ", l2ErrorNorm(phiFTCS, phiAnalyticsw))
    print("FTCS linf error norm SW = ", lInfErrorNorm(phiFTCS, phiAnalyticsw))
    print("FTBS l2 error norm SW= ", l2ErrorNorm(phiFTBS, phiAnalyticsw))
    print("FTBS linf error norm SW= ", lInfErrorNorm(phiFTBS, phiAnalyticsw))
    print("CTCS l2 error norm SW= ", l2ErrorNorm(phiCTCS, phiAnalyticsw))
    print("CTCS linf error norm SW= ", lInfErrorNorm(phiCTCS, phiAnalyticsw))
    print("BTCS l2 error norm SW= ", l2ErrorNorm(phiBTCS, phiAnalyticsw))
    print("BTCS linf error norm SW= ", lInfErrorNorm(phiBTCS, phiAnalyticsw))
    print("lax l2 error norm SW= ", l2ErrorNorm(philax, phiAnalyticsw))
    print("lax linf error norm SW= ", lInfErrorNorm(philax, phiAnalyticsw))
    

    #total variance
    #not sure how to put this in a function as it would involve reading in a function (the schemes), unless i put 
    #it in the same file as advection schemes which already has those functions defined in it! 
    # inputs to said function would be phiOld, c, nt, and outputs would be TVCTCS_tot, TVlax_tot, TVFTBS_tot
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
    
    # find the error of schemes against time
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
        
    time=np.linspace(2,nt,nt)
    
    # Plot the solutions for the cosbell function
    font = {'size'   : 20}
    plt.rc('font', **font)
    plt.figure(1)
    plt.clf()
    plt.ion()
    plt.plot(x, phiOld, label='Initial', color='black')
    plt.plot(x, phiAnalytic, label='Analytic', color='black',
             linestyle='--', linewidth=2)
    plt.plot(x, phiFTCS, label='FTCS', color='blue')
    plt.plot(x, phiFTBS, label='FTBS', color='red')
    plt.plot(x, phiCTCS, label='CTCS', color='green')
    plt.plot(x, phiBTCS, label='BTCS', color='yellow')
    plt.plot(x, philax, label='Lax Wen', color='magenta')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([-0.2,1.2])
    plt.legend(bbox_to_anchor=(1.15 , 1.1))
    plt.xlabel('$x$')
    plt.savefig('plots/cosbell.pdf', bbox_inches="tight")
    
    # Plot the solutions for the squarewave function
    fig=plt.figure(2)
    plt.clf()
    plt.ion()
    plt.plot(x, phiOldsw, label='Initial', color='black')
    plt.plot(x, phiAnalyticsw, label='Analytic', color='black',
             linestyle='--', linewidth=2)
    plt.plot(x, phiFTCSsw, label='FTCS', color='blue')
    plt.plot(x, phiFTBSsw, label='FTBS', color='red')
    plt.plot(x, phiCTCSsw, label='CTCS', color='green')
    plt.plot(x, phiBTCSsw, label='BTCS', color='yellow')
    plt.plot(x, philaxsw, label='Lax Wen', color='magenta')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([-0.2,1.2])
    plt.legend(bbox_to_anchor=(1.15 , 1.1))
    plt.xlabel('$x$')
    plt.savefig('plots/square.pdf', bbox_inches="tight")
    
    # plot the error of schemes against time
    plt.figure(3)
    plt.plot(time, phiCTCSet, label='CTCS,l2', color='green')
    plt.plot(time, philaxet, label='lax l2', color='magenta')
    plt.plot(time, phiFTCSet, label='FTCS l2', color='blue')
    plt.plot(time, phiFTBSet, label='FTBS l2', color='red')
    plt.plot(time, phiBTCSet, label='BTCS l2', color='yellow')
    plt.legend(bbox_to_anchor=(1.15 , 1.1))
    plt.xlabel('$t$')
    plt.ylabel('linf')
    plt.savefig('plots/error.pdf', bbox_inches="tight")
    
    #boundedness of lax and ctcs
    phibound=np.zeros((nt,2))
    phiboundl=np.zeros((nt,2))
    for i in range(5,nt):
        phiCTCSt = CTCS(phiOldsw.copy(), c, i)
        philaxt = lax(phiOldsw.copy(), c, i)
        phibound[i,:]=[max(phiCTCSt),min(phiCTCSt)]
        phiboundl[i,:]=[max(philaxt),min(philaxt)]
    
    plt.figure(4)
    time2=np.linspace(5,nt,nt)
    plt.plot(time2,phibound[:,0])
    plt.plot(time2,phibound[:,1])
    plt.plot(time2,phiboundl[:,0], color='magenta')
    plt.plot(time2,phiboundl[:,1], color='magenta')
    
    
    # plotting total variance
    time3=np.linspace(2,(nt-2),(nt-2))
    plt.figure(5)
    plt.plot(time3,TVCTCS_tot, label='TV CTCS')
    plt.plot(time3,TVlax_tot, label='TV lax', color='magenta')
    plt.plot(time3,TVFTBS_tot, label='TV FTBS', color='red')
    plt.xlabel('no. of time-steps')
    plt.ylabel('$TV$')
    plt.legend(bbox_to_anchor=(1.15 , 1.1))
    plt.savefig('plots/TV.pdf', bbox_inches="tight")
    
    

    input('press return to save file and continue')
    
    
    
    plt.show()

main()