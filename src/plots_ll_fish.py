import numpy as np
import matplotlib.pyplot as plt
from numpy import inf


# Part specifically for likelihood
"""helper function for ll
"""
def get_ll_alps_one(a_ind): # converts index of alpha to actual alpha value (alp_ind is number of alpha value)
    a = [0.1, 0.9, 1.1, 1.7]  # actual a values we have data for
    real_value = a[a_ind-1]
    return real_value

"""helper function for ll
"""
def get_ll_ics_one(ics_ind): # converts index of ic to actual ic value
    ics= [0.4, 0.5, 1.0, 1.3, 2.3] # actual ic values we have data for 
    real_value = ics[ics_ind-1]
    return real_value

"""read in the ll file stores in a specific format
"""
def read_ll_file(w0, n0, a, m, M, noise, path_to_file): # read in ll data file in required format
    name = "ll"+"_"+str(w0)+"_"+str(n0)+"_"+str(a)+"_"+str(m)+"_"+str(M)+"_"+str(noise)
    ending=".csv"
    csv = np.genfromtxt (path_to_file+name+ending, skip_header=1, delimiter=",")
    if M==10:
        csv[csv == -inf] = -50
    else:
        csv[csv == -inf] = -2000
    return csv

# makes single ll plot for index combination ind

def make_ll_plot(fig,ax,csv,ind, sparse):

    a_plot_nr = get_ll_alps_one(ind[0])
    ic_plot_nr = get_ll_ics_one(ind[1])
    
    # points at which ll data is evaluated
    a_x_ticks = np.arange(0.0, 2.1, 0.1)
    inits_y_ticks = np.arange(0.0, 4.1, 0.1)

    # all values below -1000 are mapped to -1000
    if sparse:
        levels = np.linspace(-50, 0, 150)
    else:
        levels = np.linspace(-2000, 0, 150)

    # mark true prm combination
    plt.plot([a_plot_nr],[ic_plot_nr],marker="x",label="Truth",color="white",markersize=12,markerfacecolor='gold',markeredgewidth=2.0, markeredgecolor="gold", zorder=10)
    
    if (np.isfinite(csv).any()):   
        ax.grid(color='grey', linestyle='-',alpha=0.1, linewidth=1)
        ax.set_facecolor('white')
        countouring=ax.contourf(a_x_ticks, inits_y_ticks, csv,30,cmap='Reds',alpha=0.9,levels=levels)
        ax.set_yticks(inits_y_ticks[::4])
        ax.set_xticks(a_x_ticks[::4])
        plt.xlabel("α")
        plt.ylabel("IC")
        cbar = fig.colorbar(countouring, fraction=0.09)
        cbar.ax.set_ylabel('Log-Likelihood')
        if sparse:
            cbar.set_ticks([-50,0])
        else:
            cbar.set_ticks([-2000, -1000, 0])
        return countouring
    else:      
        print("eieiei")
        csv[csv == -inf] = -2000
        countouring=ax.contourf(a_x_ticks, inits_y_ticks, csv,30,cmap='Reds',alpha=0.9,levels=levels)
        ax.set_yticks(inits_y_ticks[::4])
        ax.set_xticks(a_x_ticks[::4])
        plt.xlabel("α")
        plt.ylabel("IC")
        cbar = fig.colorbar(countouring,fraction=0.09)
        cbar.ax.set_ylabel('Log-Likelihood')
        if sparse:
            cbar.set_ticks([-50,0])
        else:
            cbar.set_ticks([-2000, -1000, 0])
        return("eieieie")


## Part for both likelihood and fisher
"""create plot of the bifurcation diagram
"""
def bif_plot(ax, m):
    a_vals = np.linspace(2*m, 2, 400)
    n_plus = [n(a, m, True) for a in a_vals]
    n_minus = [n(a, m, False) for a in a_vals]

    ax.plot(a_vals, np.real(n_plus), color='blue',linewidth=2)
    ax.plot(a_vals, np.real(n_minus), color='blue', linestyle="--",linewidth=2)
    ax.plot(2*m, 1, marker='o', color='blue', markersize=6)
    ax.set_ylim(0, 4)
    ax.axhline(y=0, color='blue', linewidth=4)
    ax.set_xlabel('Water Input a')
    ax.set_ylabel('Biomass n')
    ax.grid(True)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)

"""eval the biomass n equilibria in dependece of a,m (see bifurcation plot)
"""
def n(a,m,plus: bool):

    sqrt_term = np.sqrt(a**2 - 4 * m**2)

    if plus:
        return (a + sqrt_term) / (2 * m)
    else:
        return (a - sqrt_term) / (2 * m)