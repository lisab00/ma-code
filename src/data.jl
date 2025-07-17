export gen_all_ll_data, gen_all_fish_data

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
        Random.seed!(1) # make it reproducible
        df[!, "w"] .= df[!, "w"] .+ rand(Normal(0, noise), nrow(df))
        df[!, "n"] .= df[!, "n"] .+ rand(Normal(0, noise), nrow(df))
        return df
    end
end

# Functions for the likelihood analysis
"""store data
"""
function store_ll_data(w0::Float64,n0::Float64,a::Float64,m::Float64,M::Int64,noise::Float64,df::DataFrame, path_to_repo)
    CSV.write("$(path_to_repo)ll_$(w0)_$(n0)_$(a)_$(m)_$(M)_$(noise).csv", df)
end

"""computes the likelihood
"""
function compute_ll(hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=50.0)
    pred_val = sol_klausmeier(hprm; t_fixed=t_fixed, t_end=t_end)
    ll = -0.5 * sum((true_val[:,"n"] - pred_val[:,"n"]) .^2) - 0.5 * sum((true_val[:,"w"] - pred_val[:,"w"]) .^2) # add up ll for both trajectories
    #ll = -0.5 * sum((true_val[:,"n"] - pred_val[:,"n"]) .^2)
    return ll
end

"""generates and stores data for one prm combination. Run this for all a,n0,M,noise combinations wanted, helper function
"""
function gen_ll_evals_for_hprm_comb(hprm_true::Hyperprm; t_fixed::Bool=false, t_end::Float64=50.0)

    grid = create_grid()
    sol_true = sol_klausmeier(hprm_true; t_fixed=t_fixed, t_end=t_end) # returns df
    sol_true = randomize_data(sol_true, hprm_true.noise) # include noise

    ll = zeros(41, 21)

    for i in range(1, 41)
        for j in range(1, 21)
            #eval model for each point on grid
            pt = grid[i,j]
            hprm = Hyperprm(hprm_true.w0, pt[2], pt[1], hprm_true.m, hprm_true.M, hprm_true.noise) #w0,n0,a,m,M
            #eval likelihood
            ll_val = compute_ll(hprm, sol_true; t_fixed=t_fixed, t_end=t_end)
            
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
function gen_all_ll_data(index_combos, M_vals, noise_vals, m, w0, path; t_fixed::Bool=false, t_end::Float64=50.0)
    for ind in index_combos
        for M in M_vals
            for noise in noise_vals
                a_ind = ind[1]
                n0_ind = ind[2]
                hprm = Hyperprm(w0, n0_vals[n0_ind], a_vals[a_ind], m, M, noise)
                df_ll = gen_ll_evals_for_hprm_comb(hprm; t_fixed=t_fixed, t_end=t_end)
                store_ll_data(w0, n0_vals[n0_ind], a_vals[a_ind], m, M, noise, df_ll, path)
            end
        end
    end
end

# Functions for the fisher analysis
"""store data
"""
function store_fish_data(w0::Float64,m::Float64,M::Int64,noise::Float64,df::DataFrame, path_to_repo::String)
    CSV.write("$(path_to_repo)fish_$(w0)_$(m)_$(M)_$(noise).csv", df)
end

"""compute likelihood in format needed for ForwardDiff (specify variables to differentiate),same objective function
"""
function compute_ll(x, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=50.0)
    a, n0 = x
    hprm = Hyperprm(hprm.w0, n0, a, hprm.m, hprm.M, hprm.noise)
    pred_val = sol_klausmeier(hprm; t_fixed=t_fixed, t_end=t_end)
    if hprm.noise == 0.0 # then compute expected fisher info
        ll = -0.5 * sum((true_val[:,"n"] - pred_val[:,"n"]) .^2) - 0.5 * sum((true_val[:,"w"] - pred_val[:,"w"]) .^2) # add up ll for both trajectories
    else
        ll = -0.5 * 1/hprm.noise * sum((true_val[:,"n"] - pred_val[:,"n"]) .^2) - 0.5 * 1/hprm.noise * sum((true_val[:,"w"] - pred_val[:,"w"]) .^2) # add up ll for both trajectories
    end
    return ll
end

"""function that generates all the fish data needed
"""
function gen_all_fish_data(M_vals, noise_vals, m, w0, path; t_fixed::Bool=false, t_end::Float64=50.0)
    for M in M_vals
        for noise in noise_vals

            grid = create_grid()
            fish = zeros(41, 21)

            # evaluate fisher info on grid
            for i in range(1, 41)
                for j in range(1, 21)

                    pt = grid[i,j]
                    hprm = Hyperprm(w0, pt[2], pt[1], m, M, noise) #w0,n0,a,m,M

                    sol_true = sol_klausmeier(hprm; t_fixed=t_fixed, t_end=t_end) # returns df
                    sol_true = randomize_data(sol_true, hprm.noise) # include noise

                    x = [hprm.a, hprm.n0]
                    H = ForwardDiff.hessian(x -> compute_ll(x, hprm, sol_true; t_fixed=t_fixed, t_end=t_end), x)
                    FIM = -H
                    fish_val = tr(FIM)
                    fish[i,j] = fish_val
                end
            end
            
            #create data frame
            a_eval_pts = string.(0.0:0.1:2.0)
            df_fish = DataFrame(fish, a_eval_pts)

            store_fish_data(w0, m, M, noise, df_fish, path)
        end
    end
end