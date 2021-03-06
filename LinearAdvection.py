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
    nt = 100
    c = 0.2

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
    # # Calculate and print out error norms for the square wave initial conditions
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


    # find total variance of all schemes for the square wave initial condition
    TVCTCS_totsw, TVFTBS_totsw, TVlax_totsw, TVFTCS_totsw, TVBTCS_totsw = TV(phiOldsw.copy(), nt, c)

    #total variance of all schemes for cosbell initial condition
    TVCTCS_tot, TVFTBS_tot, TVlax_tot, TVFTCS_tot, TVBTCS_tot = TV(phiOld.copy(), nt, c)


    # total variance for higher resolution, cosbell
    nx_test=300
    dx_test = (xmax - xmin)/nx_test
    x_test = np.arange(xmin, xmax, dx_test)
    phiOld_test = cosBell(x_test, 0, 0.75)
    TVCTCS_tot1, TVFTBS_tot1, TVlax_tot1, TVFTCS_tot1, TVBTCS_tot1 = TV(phiOld_test.copy(), nt, c)


    #l2 error of schemes as time increases
    phiCTCSet, phiBTCSet, phiFTCSet, phiFTBSet, philaxet = l2t(phiOld.copy(), nt, c, xmax, xmin, dx, x)
    time=np.linspace(2,nt,nt)

    # calculate order of convergence for first set of resolutions
    nx_min=30
    nx_max=60
    phiFTBSex, phiCTCSex, phiFTCSex, phiBTCSex, philaxex, dx1, dx1, dx2 = order(xmin, xmax, nx_min, nx_max, c)

    # order of convergence for a finer resolutions
    nx_min1=200
    nx_max1=230
    phiFTBSex1, phiCTCSex1, phiFTCSex1, phiBTCSex1, philaxex1, dx11, dx11, dx21 = order(xmin, xmax, nx_min1, nx_max1, c)



    #stability
    # advect profiles using different value of c and vary nt inversely, to look at stability
    c_new=-0.2
    nt_new=100
    phiFTCSc = FTCS(phiOld.copy(), c_new, nt_new)
    phiFTBSc = FTBS(phiOld.copy(), c_new, nt_new)
    phiCTCSc = CTCS(phiOld.copy(), c_new, nt_new)
    phiBTCSc = BTCS(phiOld.copy(), c_new, nt_new)
    philaxc= lax(phiOld.copy(), c_new, nt_new)
    # new analytic solution
    phiAnalyticc = cosBell((x - c_new*nt_new*dx)%(xmax - xmin), 0, 0.75)

    c_new=1.053
    nt_new=19
    phiFTCSc1 = FTCS(phiOld.copy(), c_new, nt_new)
    phiFTBSc1 = FTBS(phiOld.copy(), c_new, nt_new)
    phiCTCSc1 = CTCS(phiOld.copy(), c_new, nt_new)
    phiBTCSc1= BTCS(phiOld.copy(), c_new, nt_new)
    philaxc1= lax(phiOld.copy(), c_new, nt_new)
    # new analytic solution
    phiAnalyticc1 = cosBell((x - c_new*nt_new*dx)%(xmax - xmin), 0, 0.75)

    # stability 2
    # plotting l2 error for different values of c for the schemes, to see for which values of c is the error
    # reasonable
    nt_c=50
    c_var=[-1.2,-1,-0.8,-0.5,-0.2,-0.1,0.1,0.2,0.5,0.8,1,1.2]
    errorCTCS, errorBTCS, errorFTBS, errorFTCS, errorlax = stability(phiOld.copy(), dx, xmax, xmin, x, c_var, nt_c)
    # secondly only include values of c between -1 and 1
    c_var1=[-1,-0.8,-0.5,-0.2,-0.1,0.1,0.2,0.5,0.8,1]
    errorCTCS1, errorBTCS1, errorFTBS1, errorFTCS1, errorlax1 = stability(phiOld.copy(), dx, xmax, xmin, x, c_var1, nt_c)

    #plots

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
    plt.plot(x, phiBTCS, label='BTCS', color='turquoise')
    plt.plot(x, philax, label='LW', color='magenta')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([-0.2,1.2])
    plt.legend(bbox_to_anchor=(0.95 , 1.05))
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
    plt.plot(x, phiBTCSsw, label='BTCS', color='turquoise')
    plt.plot(x, philaxsw, label='LW', color='magenta')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([-0.2,1.2])
    plt.legend(bbox_to_anchor=(0.95 , 1.05))
    plt.xlabel('$x$')
    plt.savefig('plots/square.pdf', bbox_inches="tight")

    # plot the error of schemes against time
    # not in report
    font = {'size'   : 13}
    plt.rc('font', **font)
    plt.figure(3)
    plt.plot(time, phiCTCSet, label='CTCS', color='green')
    plt.plot(time, philaxet, label='LW', color='magenta')
    plt.plot(time, phiFTCSet, label='FTCS', color='blue')
    plt.plot(time, phiFTBSet, label='FTBS', color='red')
    plt.plot(time, phiBTCSet, label='BTCS', color='turquoise')
    plt.xlabel('no. of time-steps')
    plt.ylabel('l2')
    plt.legend(bbox_to_anchor=(0.95, 1.05))
    plt.savefig('plots/error_NOTINREPORT.pdf', bbox_inches="tight")


    # plotting total variation SW initial condition
    plt.figure(6)
    plt.plot(TVCTCS_totsw, label='CTCS', color='green')
    plt.plot(TVlax_totsw, label='LW', color='magenta')
    plt.plot(TVFTBS_totsw, label='FTBS', color='red')
    plt.plot(TVFTCS_totsw, label='FTCS', color='blue')
    plt.plot(TVBTCS_totsw, label='BTCS', color='turquoise')
    plt.xlabel('no. of time-steps')
    plt.ylabel('$TV$')
    plt.ylim([1,7])
    plt.legend(bbox_to_anchor=(0.95 , 1.05))
    plt.savefig('plots/TVsw.pdf', bbox_inches="tight")

    # plotting total variation,  cosbell initial condition
    plt.figure(5)
    plt.plot(TVCTCS_tot, label='CTCS', color='green')
    plt.plot(TVlax_tot, label='LW', color='magenta')
    plt.plot(TVFTBS_tot, label='FTBS', color='red')
    plt.plot(TVFTCS_tot, label='FTCS', color='blue')
    plt.plot(TVBTCS_tot, label='BTCS', color='turquoise')
    plt.xlabel('no. of time-steps')
    plt.ylabel('$TV$')
    plt.ylim([1.7,2.1])
    plt.legend(bbox_to_anchor=(0.95 , 1.05))
    plt.savefig('plots/TVcb.pdf', bbox_inches="tight")

    # plotting total variation,  cosbell initial condition, higher resolution
    plt.figure(15)
    plt.plot(TVCTCS_tot1, label='CTCS', color='green')
    plt.plot(TVlax_tot1, label='LW', color='magenta')
    plt.plot(TVFTBS_tot1, label='FTBS', color='red')
    plt.plot(TVFTCS_tot1, label='FTCS', color='blue')
    plt.plot(TVBTCS_tot1, label='BTCS', color='turquoise')
    plt.xlabel('no. of time-steps')
    plt.ylabel('$TV$')
    plt.ylim([1.995,2.005])
    plt.legend(bbox_to_anchor=(0.95 , 1.05))
    plt.savefig('plots/TVcb1.pdf', bbox_inches="tight")

    #plot order of convergence
    font = {'size'   : 13}
    plt.rc('font', **font)
    plt.figure(7)
    # find slopes of logged error against logged dx to find order
    laxslope, laxintercept = np.polyfit(np.log(dx1), np.log(philaxex), 1)
    print("LW slope =", laxslope, "nx is", nx_min, "-", nx_max)
    ftcsslope, ftcsintercept = np.polyfit(np.log(dx1), np.log(phiFTCSex), 1)
    print("ftcs slope =",ftcsslope, "nx is", nx_min, "-", nx_max)
    ctcsslope, ctcsintercept = np.polyfit(np.log(dx1), np.log(phiCTCSex), 1)
    print("ctcs slope =", ctcsslope, "nx is", nx_min, "-", nx_max)
    btcsslope, btcsintercept = np.polyfit(np.log(dx1), np.log(phiBTCSex), 1)
    print("btcs slope =", btcsslope, "nx is", nx_min, "-", nx_max)
    ftbsslope, ftbsintercept = np.polyfit(np.log(dx1), np.log(phiFTBSex), 1)
    print("ftbs slope =", ftbsslope, "nx is", nx_min, "-", nx_max)
    plt.plot(dx1,phiCTCSex, label='CTCS, slope = %.2f'%(ctcsslope), color="green", marker="*")
    plt.plot(dx1,phiBTCSex, label='BTCS, slope = %.2f'%(btcsslope), color="turquoise", marker="x")
    plt.plot(dx1,phiFTCSex, label='FTCS, slope = %.2f'%(ftcsslope), color="blue", marker="v")
    plt.plot(dx1,phiFTBSex, label='FTBS, slope =%.2f'%(ftbsslope), color="red", marker="o")
    plt.plot(dx1,philaxex, label='LW, slope =%.2f'%(laxslope), color="magenta", marker=".")
    plt.plot(dx1,dx1, label='$\Delta x$', color="orange", linestyle="--")
    plt.plot(dx1,dx2, label='$\Delta x^2$', color="brown", linestyle="--")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$\Delta x$')
    plt.ylabel('l2')
    plt.title('Order of convergence for nx = %.0f to %.0f'% (nx_min, nx_max))
    plt.legend(bbox_to_anchor=(0.95 , 1.05))
    plt.savefig('plots/order.pdf', bbox_inches="tight")
    #print the slopes of the error against dx


    # second plot for order of convergence, different set of resolutions
    plt.figure(8)
    laxslope1, laxintercept1 = np.polyfit(np.log(dx11), np.log(philaxex1), 1)
    print("LW slope =", laxslope1, "nx is", nx_min1, "-", nx_max1)
    ftcsslope1, ftcsintercept1 = np.polyfit(np.log(dx11), np.log(phiFTCSex1), 1)
    print("ftcs slope =",ftcsslope1, "nx is", nx_min1, "-", nx_max1)
    ctcsslope1, ctcsintercept1 = np.polyfit(np.log(dx11), np.log(phiCTCSex1), 1)
    print("ctcs slope =", ctcsslope1, "nx is", nx_min1, "-", nx_max1)
    btcsslope1, btcsintercept1 = np.polyfit(np.log(dx11), np.log(phiBTCSex1), 1)
    print("btcs slope =", btcsslope1, "nx is", nx_min1, "-", nx_max1)
    ftbsslope1, ftbsintercept1 = np.polyfit(np.log(dx11), np.log(phiFTBSex1), 1)
    print("ftbs slope =", ftbsslope1, "nx is", nx_min1, "-", nx_max1)
    plt.plot(dx11,phiCTCSex1, label='CTCS, slope = %.2f'%(ctcsslope1), color="green", marker="*")
    plt.plot(dx11,phiBTCSex1, label='BTCS, slope = %.2f'%(btcsslope1), color="turquoise", marker="x")
    plt.plot(dx11,phiFTCSex1, label='FTCS, slope = %.2f'%(ftcsslope1), color="blue", marker="v")
    plt.plot(dx11,phiFTBSex1, label='FTBS, slope = %.2f'%(ftbsslope1), color="red", marker="o")
    plt.plot(dx11,philaxex1, label='LW, slope = %.2f'%(laxslope1), color="magenta", marker=".")
    plt.plot(dx11,dx11, label='$\Delta x$', color="orange", linestyle="--")
    plt.plot(dx11,dx21, label='$\Delta x^2$', color="brown", linestyle="--")
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks([5.7e-4,6.3e-4])
    plt.xlabel('$\Delta x$')
    plt.ylabel('l2')
    plt.title('Order of convergence for nx = %.0f to %.0f'% (nx_min1, nx_max1))
    plt.legend(bbox_to_anchor=(0.95 , 1.05))
    plt.savefig('plots/order1.pdf', bbox_inches="tight")

    #plots for stability, c=-1.2
    font = {'size'   : 20}
    plt.rc('font', **font)
    plt.figure(9)
    plt.plot(x, phiOld, label='Initial', color='black')
    plt.plot(x, phiAnalyticc, label='Analytic', color='black',
             linestyle='--', linewidth=2)
    plt.plot(x, phiFTCSc, label='FTCS', color='blue')
    plt.plot(x, phiFTBSc, label='FTBS', color='red')
    plt.plot(x, phiCTCSc, label='CTCS', color='green')
    plt.plot(x, phiBTCSc, label='BTCS', color='turquoise')
    plt.plot(x, philaxc, label='LW', color='magenta')
    plt.axhline(0, linestyle=':', color='black')
    plt.legend(bbox_to_anchor=(0.95 , 1.05))
    plt.ylim([-0.2,1.2])
    plt.xlabel('$x$')
    plt.savefig('plots/stability.pdf', bbox_inches="tight")

    # stability plot, c is 1.053
    font = {'size'   : 20}
    plt.rc('font', **font)
    plt.figure(14)
    plt.plot(x, phiOld, label='Initial', color='black')
    plt.plot(x, phiAnalyticc1, label='Analytic', color='black',
             linestyle='--', linewidth=2)
    plt.plot(x, phiFTCSc1, label='FTCS', color='blue')
    plt.plot(x, phiFTBSc1, label='FTBS', color='red')
    plt.plot(x, phiCTCSc1, label='CTCS', color='green')
    plt.plot(x, phiBTCSc1, label='BTCS', color='turquoise')
    plt.plot(x, philaxc1, label='LW', color='magenta')
    plt.axhline(0, linestyle=':', color='black')
    plt.legend(bbox_to_anchor=(0.95 , 1.05))
    plt.ylim([-0.2,1.2])
    plt.xlabel('$x$')
    plt.savefig('plots/stability1.pdf', bbox_inches="tight")

    # plots for stability 2
    # error for different values of c including c values outside [-1,1]
    plt.figure(10)
    plt.plot(c_var, errorCTCS, "o", label='CTCS', color='green')
    plt.plot(c_var, errorlax, "v", label='LW', color='magenta')
    plt.plot(c_var, errorFTCS, "x", label='FTCS', color='blue')
    plt.legend(bbox_to_anchor=(0.95 , 1.05))
    plt.xlabel('$c$')
    plt.ylabel('$l2$')
    plt.savefig('plots/stabilityctcslax.pdf', bbox_inches="tight")

    # plots for # error for different values of c between -1 and 1
    plt.figure(11)
    plt.plot(c_var1, errorCTCS1, label='CTCS', color='green')
    plt.plot(c_var1, errorlax1, label='LW', color='magenta')
    #plt.plot(c_var1, errorFTCS1, "x", label='FTCS', color='blue')
    plt.legend(bbox_to_anchor=(0.95 , 1.05))
    plt.xlabel('$c$')
    plt.ylabel('$l2$')
    plt.savefig('plots/stabilityctcslax2.pdf', bbox_inches="tight")




    plt.show()
main()
