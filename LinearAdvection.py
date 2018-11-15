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
    nx = 60
    nt = 80
    c = 0.2

    # Derived parameters
    dx = (xmax - xmin)/nx
    print(dx**2)
    print(dx)
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
    TVCTCS_tot, TVFTBS_tot, TVlax_tot = TV(phiOld.copy(), nt, c)   
    # find the l2 error of schemes at different times
    phiCTCSet, phiBTCSet, phiFTCSet, phiFTBSet, philaxet = l2t(phiOld.copy(), nt, c, xmax, xmin, dx, x)
    time=np.linspace(2,nt,nt)
    # order of convergence
    nx_min=30
    nx_max=60
    phiFTBSex, phiCTCSex, phiFTCSex, phiBTCSex, philaxex, dx2, dx21, dx32 = order(xmin, xmax, nx_min, nx_max, c) 
    
    
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
    #plt.plot(time, phiFTCSet, label='FTCS l2', color='blue')
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
    # lax better than ctcs cos TV is less
    
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
    
    #plot order of convergence
    plt.figure(6)
    plt.plot(dx2,phiCTCSex, label='CTCS,l2', color="green", marker="*")
    plt.plot(dx2,phiBTCSex, label='BTCS,l2', color="red", marker="x")
    plt.plot(dx2,phiFTCSex, label='FTCS,l2', color="blue", marker="v")
    plt.plot(dx2,phiFTBSex, label='FTBS,l2', color="yellow", marker="o")
    plt.plot(dx2,philaxex, label='lax,l2', color="magenta", marker=".")
    plt.plot(dx2,dx21, label='$\Delta x$', color="orange", linestyle="--")
    plt.plot(dx2,dx32, label='$\Delta x^2$', color="brown", linestyle="--")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$\Delta x$')
    plt.ylabel('l2')
    plt.legend(bbox_to_anchor=(1.15 , 1.1))
    plt.savefig('plots/order.pdf', bbox_inches="tight")

    input('press return to save file and continue')
    
    
    
    plt.show()

main()
