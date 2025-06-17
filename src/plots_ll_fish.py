import numpy as np
import matplotlib.pyplot as plt
import math

from numpy import inf

# reference: https://github.com/ElisabethRoesch/Bifurcations  

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

"""read in the ll file stored in a specific format
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

"""makes single ll plot for index combination ind
"""
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
    
"""generate all likelihood plots
"""
def make_all_ll_plots(index_combos, M_vals, noise_vals, m, w0, path_to_read, path_to_store, store=False):

    for ind in index_combos:
        for M in M_vals:
            for noise in noise_vals:

                if M==10:
                    sparse = True
                else:
                    sparse = False

                n0 = get_ll_ics_one(ind[1])
                a = get_ll_alps_one(ind[0])

                csv = read_ll_file(w0,n0,a,m,M,noise,path_to_read)

                fig, ax = plt.subplots()
                make_ll_plot(fig, ax, csv, ind, sparse)
                ax.set_title(f"M={M}, noise={noise}")
                bif_plot(ax,m)

                if store:
                    plt.savefig(f"{path_to_store}ll_{w0}_{n0}_{a}_{m}_{M}_{noise}.pdf", bbox_inches='tight')


## Part specifically for fisher
"""read in the fish file stored in a specific format
"""
def read_fish_file(w0, m, M, noise, path_to_file):
    name="fish"+"_"+str(w0)+"_"+str(m)+"_"+str(M)+"_"+str(noise)
    ending=".csv"
    csv = np.genfromtxt (path_to_file+name+ending, skip_header=1, delimiter=",")
    return csv

"""create single fisher plot on whole prm grid
"""
def make_fish_plot(fig, ax, csv):

    min_val = np.min(csv)
    if min_val <= 0:
        csv = csv - min_val + 1e-10  # Shift all values so the minimum becomes ~0+

    # take log data bec values are very high
    csv = np.log(csv)

    # points at which fish data is evaluated
    a_x_ticks = np.arange(0.0, 2.1, 0.1)
    inits_y_ticks = np.arange(0.0, 4.1, 0.1)

    # all values below -8 are mapped to -8
    levels = np.linspace(math.ceil(np.min(csv)),  math.floor(np.max(csv)),20)

    #ax.grid(color='grey', linestyle='-',alpha=0.1, linewidth=1)
    #ax.set_facecolor('white')
    countouring=ax.contourf(a_x_ticks, inits_y_ticks, csv,30,cmap='Blues',alpha=0.9, levels=levels)
    ax.set_yticks(inits_y_ticks[::4])
    ax.set_xticks(a_x_ticks[::4])
    ax.set_xlabel("a")
    ax.set_ylabel("IC")
    cbar = fig.colorbar(countouring,fraction=0.09)
    cbar.ax.set_ylabel('Fisher information')
    cbar.set_ticks([math.ceil(np.min(csv)), math.floor(np.max(csv))])
    return countouring

"""create fisher plots for all parameter combinations
"""
def make_all_fish_plots(M_vals, noise_vals, w0, m, path_to_read, path_to_store, store=False):

    for M in M_vals:
        for noise in noise_vals:
            
            csv = read_fish_file(w0,m,M,noise,path_to_read)
            fig, ax = plt.subplots()
            make_fish_plot(fig, ax, csv)
            bif_plot(ax,m)
            ax.set_title(f"M={M}, noise={noise}")

            if store: 
                plt.savefig(f"{path_to_store}fish_{w0}_{m}_{M}_{noise}.pdf", bbox_inches='tight')

"""plots the marginal fisher information, evaluated at every a and summed/ averaged across all ICs
"""
def fi_avg_ic(fig, ax, csv):
    plt.xlabel("a")
    a_x_ticks = np.arange(0.0, 2.1, 0.1)
    ax.set_xticks(a_x_ticks[::4])
    ax.set_title(f"Marginal Fisher Information averaged across all n0")

    finit_sums=[]    
    for j in range(len(csv[1])): #21: for each a
        j_sum = 0
        for i in range(len(csv)): #41: sum across all ics
            if np.isfinite(csv[i][j]):
                j_sum+=csv[i][j]
        finit_sums.append(j_sum)

    log_sums = np.log(finit_sums)
    ic_side = ax.plot(a_x_ticks,log_sums)

    return ic_side

"""plots the marginal fisher information, evaluated at every a and a given IC
"""
def fi_ic(fig, ax, csv, ic):
    plt.xlabel("a")

    inits_y_ticks = np.arange(0.0, 4.1, 0.1)
    print(np.where(inits_y_ticks == ic))
    ind = np.where(np.isclose(inits_y_ticks, ic))[0]
    a_x_ticks = np.arange(0.0, 2.1, 0.1)
    ax.set_xticks(a_x_ticks[::4])
    ax.set_title(f"Marginal Fisher Information for n0={ic}")
    log_fish = np.log(csv[ind[0]])
    
    ic_side = ax.plot(a_x_ticks,log_fish)

    return ic_side


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