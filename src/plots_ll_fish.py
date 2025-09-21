import numpy as np
import matplotlib.pyplot as plt
import math

# reference: https://github.com/ElisabethRoesch/Bifurcations  

# Part specifically for likelihood
def get_a_from_index(a_ind): 
    """
    helper function for ll. Converts index of alpha to actual alpha value (alp_ind is number of alpha value)

    Args:
        a_ind (int): index of parameter a to be considered (must be between 1 and 5)

    Returns:
        actual value for parameter a
    """
    a = [0.1, 0.9, 1.1, 1.3, 1.7, 1.9, 0.8] # actual a values we have data for
    return a[a_ind-1]


def get_ic_from_index(ics_ind):
    """
    helper function for ll. Converts index of ic to actual ic value (ics_ind is number of ic value)

    Args:
        a_ind (int): index of parameter a to be considered (must be between 1 and 5)

    Returns:
        actual value for parameter a
    """
    ics= [0.2, 0.4, 1.0, 1.3, 2.3] # actual ic values we have data for
    return ics[ics_ind-1]


def read_ll_file(w0, n0, a, m, M, noise, path_to_file):
    """
    read in the ll file stored in a specific format.
    """
    name = "ll"+"_"+str(w0)+"_"+str(n0)+"_"+str(a)+"_"+str(m)+"_"+str(M)+"_"+str(noise)
    ending=".csv"
    csv = np.genfromtxt (path_to_file+name+ending, skip_header=1, delimiter=",")
    return csv


def make_ll_plot(fig,ax,csv,ind,lower_bound):
    """
    make single ll plot for index combination ind

    Args:
        `csv`: returned by read_ll_file
        `ind`: index of true parameter point
        `lower_bound`: lower bound where ll values are cut off
    """
    a = get_a_from_index(ind[0])
    n0 = get_ic_from_index(ind[1])
    
    # points at which ll data was evaluated
    a_x_ticks = np.arange(0.0, 2.1, 0.1)
    n0_y_ticks = np.arange(0.0, 4.1, 0.1)

    # all values below are mapped to lower_bound
    levels = np.linspace(lower_bound, 0, 150)

    contouring=ax.contourf(a_x_ticks, n0_y_ticks, csv,30,cmap='Reds',alpha=0.9,levels=levels)

    # mark true prm combination
    plt.plot([a],[n0],marker="x",label="true",linestyle="None",markersize=12,markerfacecolor='gold',markeredgewidth=2.0, markeredgecolor="gold", zorder=10)
    #ax.grid(color='grey', linestyle='-',alpha=0.1, linewidth=1)
    ax.set_facecolor('white')
    ax.legend()
    ax.set_xticks(np.arange(0, 2.1, 1.0))  # Only show 0.0, 1.0, 2.0
    ax.set_yticks(np.arange(2.0, 4.1, 2.0))  # Only show 0.0, 2.0, 4.0
    plt.xlabel("Water input parameter")
    plt.ylabel("Biomass IC")
    cbar = fig.colorbar(contouring, fraction=0.09)
    cbar.ax.set_ylabel('Log-Likelihood')
    cbar.set_ticks([lower_bound, 0])
        
    return contouring
    

def make_all_ll_plots(index_combos, M_vals, noise_vals, m, w0, t_fixed, lower_bound, path_to_read, path_to_store, store=False):
    """
    create (and store) log-likelihood plots for several points.

    Args:
        `index_combos`: indices of points whose log-likelihood should be plotted
        `M_vals`: M values for which ll should be plotted
        `noise_vals`: noise values for which ll should be plotted
        `path_to_read`: path to folder where ll file is stored
        `path_to_store`: path to folder where plot should be stored
        `store`: set True if plot should be saved as .pdf file
        `t_fixed`: True if observation time window is fixed
    """
    for ind in index_combos:
        for M in M_vals:
            for noise in noise_vals:

                n0 = get_ic_from_index(ind[1])
                a = get_a_from_index(ind[0])

                csv = read_ll_file(w0,n0,a,m,M,noise,path_to_read)

                fig, ax = plt.subplots()

                make_ll_plot(fig, ax, csv, ind, lower_bound)
                ax.set_title(f"M={M}, noise={noise}, t_fixed={t_fixed}")
                #bif_plot(ax,m)

                if store:
                    plt.savefig(f"{path_to_store}ll_{w0}_{n0}_{a}_{m}_{M}_{noise}.pdf", bbox_inches='tight')


def ll_grid_plot(ind, noise_vals, M_vals, lower_bound, path_to_read, w0,m):
    """
    plot 3x3 grid with log-likelihood plots for given point.
    M decreases from left to right, noise increases from top to bottom.as_integer_ratio

    Args:
        `ind`: index of true parameter point
        `noise_vals`: 3 noise levels, increasing order
        `M_vals`: 3 M values, decreasing order
        `lower_bound`: lower bound where ll values are cut off
        `path_to_read`: path to folder where the csv with the ll values is stored

    Returns:
        3x3 grid of plots
    """
    n0 = get_ic_from_index(ind[1])
    a = get_a_from_index(ind[0])

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    
    # points at which ll data was evaluated
    a_x_ticks = np.arange(0.0, 2.1, 0.1)
    n0_y_ticks = np.arange(0.0, 4.1, 0.1)

    # all values below are mapped to lower_bound
    levels = np.linspace(lower_bound, 0, 150)

    for i in range(0, len(noise_vals)):
        for j in range(0, len(M_vals)):

            ax = axes[i,j]
            csv = read_ll_file(w0,n0,a,m,M_vals[j],noise_vals[i],path_to_read)
            contouring=ax.contourf(a_x_ticks, n0_y_ticks, csv,30,cmap='Reds',alpha=0.9,levels=levels)
            ax.plot([a],[n0],marker="x",linestyle="None",markersize=12,markerfacecolor='gold',markeredgewidth=2.0, markeredgecolor="gold", zorder=10)
            ax.set_xticks(np.arange(0, 2.1, 1.0))  # Only show 0.0, 1.0, 2.0
            ax.set_yticks(np.arange(2.0, 4.1, 2.0))  # Only show 0.0, 2.0, 4.0
            
    cbar = fig.colorbar(contouring, ax=axes, orientation='vertical', fraction=0.04, pad=0.04)
    cbar.set_label("Log-likelihood")
    cbar.set_ticks([lower_bound, lower_bound/2, 0])

    # Add single figure-wide legend
    fig.legend(["true"], loc='lower center', ncol=1, bbox_to_anchor=(0.9, 0.85))



## Part specifically for fisher
def read_fish_file(w0, m, M, noise, path_to_file):
    """
    read in the fish file stored in a specific format
    """
    name="fish"+"_"+str(w0)+"_"+str(m)+"_"+str(M)+"_"+str(noise)
    ending=".csv"
    csv = np.genfromtxt (path_to_file+name+ending, skip_header=1, delimiter=",")
    return csv


def make_fish_plot(fig, ax, csv, log=True):
    """
    create single fisher plot on whole prm grid

    Args:
        `csv`: returned by read_fish_file
        `log`: if True logarithm is applied to data (when values are high)
    """
    if log:
        #min_val = np.min(csv)
        #if min_val <= 0:
         #   csv = csv - min_val + 1e-10  # Shift all values so the minimum becomes ~0+
         
        # Find the smallest strictly positive value
        positive_min = np.min(csv[csv > 0])

        # Replace all negative (and zero) values with positive_min
        csv = np.where(csv <= 0, positive_min, csv)

        # take log data bec values are very high
        csv = np.log(csv)

    # points at which fish data is evaluated
    a_x_ticks = np.arange(0.0, 2.1, 0.1)
    inits_y_ticks = np.arange(0.0, 4.1, 0.1)

    # all values below -8 are mapped to -8
    levels = np.linspace(math.floor(np.min(csv)),  math.ceil(np.max(csv)),150)

    #ax.grid(color='grey', linestyle='-',alpha=0.1, linewidth=1)
    #ax.set_facecolor('white')
    contouring=ax.contourf(a_x_ticks, inits_y_ticks, csv,30,cmap='Blues',alpha=0.9, levels=levels)
    ax.set_yticks(inits_y_ticks[::4])
    ax.set_xticks(a_x_ticks[::4])
    ax.set_xlabel("Water input parameter")
    ax.set_ylabel("Biomass IC")
    cbar = fig.colorbar(contouring,fraction=0.09)
    cbar.ax.set_ylabel('Fisher information')
    cbar.set_ticks([math.floor(np.min(csv)), math.ceil(np.max(csv))])
    return contouring


def make_all_fish_plots(M_vals, noise_vals, w0, m, path_to_read, path_to_store, t_fixed, store=False, log=True):
    """
    create (and store) fisher plots for all parameter combinations.

    Args:
        M_vals: M values for which fi should be plotted
        noise_vals: noise values for which fi should be plotted
        path_to_read: path to folder where data is stored
        path_to_store: path to folder where plot should be stored
        t_fixed: True if observation time window is fixed
        store: set True if plot should be saved as pdf
        log: if True logarithm is applied to data (when values are high)
    """
    for M in M_vals:
        for noise in noise_vals:
            
            csv = read_fish_file(w0,m,M,noise,path_to_read)
            fig, ax = plt.subplots()
            make_fish_plot(fig, ax, csv, log)
            #bif_plot(ax,m)
            ax.set_title(f"M={M}, noise={noise}, t_fixed={t_fixed}")

            if store: 
                plt.savefig(f"{path_to_store}fish_{w0}_{m}_{M}_{noise}.pdf", bbox_inches='tight')


def fish_grid_plot(noise_vals, M_vals, path_to_read, w0,m,log=True):
    """
    plot 3x3 grid with Fisher information plots
    M decreases from left to right, noise increases from top to bottom.as_integer_ratio

    Args:
        `noise_vals`: 3 noise levels, increasing order
        `M_vals`: 3 M values, decreasing order
        `lower_bound`: lower bound where ll values are cut off
        `path_to_read`: path to folder where the csv with the ll values is stored
        `log`: set True to apply log to data (if values are very high)
        
    Returns:
        3x3 grid of plots
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    
    # points at which fish data was evaluated
    a_x_ticks = np.arange(0.0, 2.1, 0.1)
    n0_y_ticks = np.arange(0.0, 4.1, 0.1)
    upper_bounds = []

    for i in range(0, len(noise_vals)):
        for j in range(0, len(M_vals)):
            csv = read_fish_file(w0,m,M_vals[j],noise_vals[i],path_to_read)

            if log:
                #min_val = np.min(csv)
                #if min_val <= 0:
                 #   csv = csv - min_val + 1e-10  # Shift all values such that the minimum becomes 0+
                
                # Find the smallest strictly positive value
                positive_min = np.min(csv[csv > 0])

                # Replace all negative (and zero) values with positive_min
                csv = np.where(csv <= 0, positive_min, csv)
                
                # take log data because values are very high
                csv = np.log(csv)
    
            upper_bounds.append(math.ceil(np.max(csv)))

    upper_bound = max(upper_bounds)
    
    # all values above are mapped to upper_bound
    levels = np.linspace(0, upper_bound, 150)

    for i in range(0, len(noise_vals)):
        for j in range(0, len(M_vals)):

            csv = read_fish_file(w0,m,M_vals[j],noise_vals[i],path_to_read)
            min_val = np.min(csv)
            if min_val <= 0:
                csv = csv - min_val + 1e-10  # Shift all values so the minimum becomes ~0+

            # take log data bec values are very high
            if log:
                csv = np.log(csv)

            ax = axes[i,j]
            contouring=ax.contourf(a_x_ticks, n0_y_ticks, csv,30,cmap='Blues',alpha=0.9,levels=levels)
            ax.set_xticks(np.arange(0, 2.1, 1.0))  # Only show 0.0, 1.0, 2.0
            ax.set_yticks(np.arange(2.0, 4.1, 2.0))  # Only show 0.0, 2.0, 4.0
            
    cbar = fig.colorbar(contouring, ax=axes, orientation='vertical', fraction=0.04, pad=0.04)
    cbar.set_label("Fisher information")
    cbar.set_ticks([0, upper_bound])


def fi_avg_ic(fig, ax, csv):
    """
    plots the average marginal fisher information.
    I.e., a horizontal section in the variable a, for each a showing the average FI across the vertical n0 axis.

    Args:
        csv: output of read_fish_file
    """
    plt.xlabel("a")
    a_x_ticks = np.arange(0.0, 2.1, 0.1)
    ax.set_xticks(a_x_ticks[::4])
    ax.set_title(f"Marginal Fisher information averaged across all n0")

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


def fi_ic(fig, ax, csv, ic):
    """
    plots the marginal fisher information for a given n0. 
    I.e., a horizontal section in the variable a, showing the FI for all n0 values.

    Args:
        csv: output of read_fish_file
        ic: n0 for which marginal FI should be plotted
    """
    plt.xlabel("a")

    inits_y_ticks = np.arange(0.0, 4.1, 0.1)
    print(np.where(inits_y_ticks == ic))
    ind = np.where(np.isclose(inits_y_ticks, ic))[0]
    a_x_ticks = np.arange(0.0, 2.1, 0.1)
    ax.set_xticks(a_x_ticks[::4])
    ax.set_title(f"Marginal Fisher information for n0={ic}")
    log_fish = np.log(csv[ind[0]])
    
    ic_side = ax.plot(a_x_ticks,log_fish)

    return ic_side


## Part for both likelihood and fisher
def bif_plot(ax, m):
    """
    create plot of the bifurcation diagram. Plot water input parameter a against biomass equilibria, indicating stability of the branches.

    Args:
        m: mortality rate of biomass compartment
    """
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


def n(a,m,plus: bool):
    """
    eval the biomass n equilibria in dependece of a,m (see bifurcation plot).

    Args:
        a: water input parameter
        m: plant mortality parameter
        plus: true, if upper branch of fold bifurcation should be plotted
    """
    sqrt_term = np.sqrt(a**2 - 4 * m**2)

    if plus:
        return (a + sqrt_term) / (2 * m)
    else:
        return (a - sqrt_term) / (2 * m)