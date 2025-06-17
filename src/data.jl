export gen_all_ll_data

"""create the grid on which the functions shall be evaluated
we evaluate likelihood for a in (0,2), n0 in (0,4) on an uniformly spaced grid
"""
function create_grid()
    a_vals = 0.0:0.1:2.0
    n_vals = 0.0:0.1:4.0

    grid = [(a, n) for n in n_vals, a in a_vals] 
    return grid
end

"""add Gaussian noise to simulated data
"""
function randomize_data(df::DataFrame, noise::Float64)
    if noise == 0.0
        return df
    else
        Random.seed!(1)
        df[!, "w"] .= df[!, "w"] .+ rand(Normal(0, noise), nrow(df))
        df[!, "n"] .= df[!, "n"] .+ rand(Normal(0, noise), nrow(df))
        return df
    end
end

# Functions for the likelihood analysis
"""store data
"""
function store_ll_data(w0::Float64,n0::Float64,a::Float64,m::Float64,M::Int64,noise::Float64,df::DataFrame, path_to_repo)
    CSV.write("$(path_to_repo)ma-code/data/likelihood/m0.45/ll_$(w0)_$(n0)_$(a)_$(m)_$(M)_$(noise).csv", df)
end

"""computes the likelihood
"""
function compute_ll(hprm::Hyperprm, true_val::DataFrame)
    pred_val = sol_klausmeier(hprm)
    ll = -0.5 * sum((true_val[:,"n"] - pred_val[:,"n"]) .^2) - 0.5 * sum((true_val[:,"w"] - pred_val[:,"w"]) .^2) # add up ll for both trajectories
    #ll = -0.5 * sum((true_val[:,"n"] - pred_val[:,"n"]) .^2)
    return ll
end

"""generates and stores data for one prm combination. Run this for all a,n0,M,noise combinations wanted, helper function
"""
function gen_ll_evals_for_hprm_comb(hprm_true::Hyperprm)

    grid = create_grid()
    sol_true = sol_klausmeier(hprm_true) # returns df
    sol_true = randomize_data(sol_true, hprm_true.noise) # include noise

    ll = zeros(41, 21)

    for i in range(1, 41)
        for j in range(1, 21)
            #eval model for each point on grid
            pt = grid[i,j]
            hprm = Hyperprm(hprm_true.w0, pt[2], pt[1], hprm_true.m, hprm_true.M, hprm_true.noise) #w0,n0,a,m,M
            #eval likelihood
            ll_val = compute_ll(hprm, sol_true)
            
            ll[i,j] = ll_val
        end
    end
    
    #return data frame
    a_eval_pts = string.(0.0:0.1:2.0)
    df_ll = DataFrame(ll, a_eval_pts)

    return df_ll
end

"""function that generates all the ll data needed
"""
function gen_all_ll_data(index_combos, M_vals, noise_vals, m, w0, path)
    for ind in index_combos
        for M in M_vals
            for noise in noise_vals
                a_ind = ind[1]
                n0_ind = ind[2]
                hprm = Hyperprm(w0, n0_vals[n0_ind], a_vals[a_ind], m, M, noise)
                df_ll = gen_ll_evals_for_hprm_comb(hprm)
                store_ll_data(w0, n0_vals[n0_ind], a_vals[a_ind], m, M, noise, df_ll, path)
            end
        end
    end
end