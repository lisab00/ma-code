export gen_all_ll_data, gen_all_fish_data


# Tools
"""
    function create_grid()

Create the evaluation grid for a function. We evaluate for a in (0,2), n0 in (0,4) on a uniformly spaced grid with mesh size 0.1
"""
function create_grid()
    a_vals = 0.0:0.1:2.0
    n_vals = 0.0:0.1:4.0

    grid = [(a, n) for n in n_vals, a in a_vals] 
    return grid
end

"""
    function randomize_data!(df::DataFrame, noise::Float64)

add mean-zero Gaussian noise to simulated data.

# Arguments
- `df::DataFrame`: data to add noise on
- `noise::Float64`: noise level sigma^2 (i.e. variance of Gaussian)

# Returns
-`DataFrame`: with randomized data
"""
function randomize_data!(df::DataFrame, noise::Float64)
    if noise == 0.0
        return df
    else
        df[!, "w"] .= df[!, "w"] .+ rand(Normal(0, noise), nrow(df))
        df[!, "n"] .= df[!, "n"] .+ rand(Normal(0, noise), nrow(df))
        return df
    end
end


# Maximum likelihood estimation
"""
    function compute_mle(hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)

compute the maximum likelihood estimate given data observations by minimizing the negative log-likelihood function using the Optim.jl package.
The initialization point is chosen as the true parameter combination underlying the data observation to ensure fast convergence to global minimum.
The minimization method is chosen by default.

# Returns
- `Vector{Float64}`: 2-element vector containing the mle [a_mle, n0_mle]
- `Bool`: true if optimization was successfull
"""
function compute_mle(hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0, N::Int64=5)
    inits, inits_loss, mles, losses, best_loss_ind, converged = mult_restart_mle(N, hprm, true_val; t_fixed=t_fixed, t_end=t_end, t_step=t_step)
    return mles[best_loss_ind, :], converged[best_loss_ind]
end

"""
    function mult_restart_mle(N::Int64, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)

Perform Maximum Likelihood estimation for N different starting points. Goal is to find global minimum

# Inputs
    - `N::Int64`: number of restarts

# Returns
    - `Maxtrix`: initial values used in optimization
    - `Matrix`: computed MLEs
    - `Vector`: corresponding losses of MLEs
"""
function mult_restart_mle(N::Int64, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)
    # generate optim start pts
    inits = hcat(2 .* rand(N), 4 .* rand(N))

    # store mles and corresponding loss
    mle_vals = zeros(N, 2)
    mle_loss, inits_loss, converged = zeros(N), zeros(N), zeros(N)

    for i in 1:N
        pt = inits[i,:]
        result = optimize(x -> - compute_ll(x, hprm, true_val; t_fixed=t_fixed, t_end=t_end, t_step=t_step), pt)
        #display(result)
        mle_vals[i,:] = Optim.minimizer(result)
        mle_loss[i] =  Optim.minimum(result)
        converged[i] = Optim.converged(result)
        inits_loss[i] = -compute_ll(pt, hprm, true_val; t_fixed=t_fixed, t_end=t_end, t_step=t_step)
    end

    # extract best
    best_loss, best_loss_ind = findmin(mle_loss)

    return inits, inits_loss, mle_vals, mle_loss, best_loss_ind, converged
end


# Functions for the likelihood analysis
"""
    function store_ll_data(w0::Float64,n0::Float64,a::Float64,m::Float64,M::Int64,noise::Float64,df::DataFrame, path_to_repo)

stores data evaluated on grid in a csv file.
Name of form "ll_w0_n0_a_m_M_noise.csv"

# Arguments
- `df::DataFrame`: df to store
- `path_to_repo::String`: path to folder where to store the file
"""
function store_ll_data(w0::Float64,n0::Float64,a::Float64,m::Float64,M::Int64,noise::Float64,df::DataFrame, path_to_store::String)
    CSV.write("$(path_to_store)ll_$(w0)_$(n0)_$(a)_$(m)_$(M)_$(noise).csv", df)
end

"""
    function compute_ll(x, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)

compute the log-likelihood in least-squares form for Klausmeier model for data with Gaussian noise. First, simulate Klausmeier model for given hyperparameters and noise level. Then, compare to true trajectories.
Includes x variables needed for ForwardDiff and Optim.

# Arguments
- `x`: variables with respect to which is differentiated
- `hprm::Hyperprm`: parameters for which the Klausmeier simulation is performed
- `true_val::DataFrame`: true data trajectories. DataFrame with columns "w" and "n".
- `t_fixed::Bool`: true if we consider a fixed observation time window
- `t_end::Float64`: end of observation window (if t_fixed=true)
- `t_step::Float64`: step size with which M observations should be picked (set if t_fixed=false)

# Returns
- `Float`: scalar value of log-likelihood at given grid point 
"""
function compute_ll(x, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)
    a, n0 = x
    hprm = Hyperprm(hprm.w0, n0, a, hprm.m, hprm.M, hprm.noise)
    pred_val = sol_klausmeier(hprm; t_fixed=t_fixed, t_end=t_end, t_step=t_step)
    if hprm.noise == 0.0
        ll = -0.5 * sum((true_val[:,"n"] - pred_val[:,"n"]) .^2) - 0.5 * sum((true_val[:,"w"] - pred_val[:,"w"]) .^2) # add up ll for both trajectories
    else
        ll = -0.5 * 1/hprm.noise * sum((true_val[:,"n"] - pred_val[:,"n"]) .^2) - 0.5 * 1/hprm.noise * sum((true_val[:,"w"] - pred_val[:,"w"]) .^2) # add up ll for both trajectories
    end
    return ll
end

"""
    function gen_ll_evals_for_hprm_comb(hprm_true::Hyperprm; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)

evaluates log-likelihood on grid for one (a,n0,M,noise) hyperprm combination. Run this for all hyperprm combinations wanted, helper function

# Arguments
- `hprm::Hyperprm`: parameters for which the Klausmeier simulation is performed
- `true_val::DataFrame`: true data trajectories. DataFrame with columns "w" and "n".
- `t_fixed::Bool`: true if we consider a fixed observation time window
- `t_end::Float64`: end of observation window (if t_fixed=true)
- `t_step::Float64`: step size with which M observations should be picked (set if t_fixed=false)

# Returns
-`DataFrame`: DataFrame of log-likelihood evaluated on grid for given parameter combination
"""
function gen_ll_evals_for_hprm_comb(hprm_true::Hyperprm; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)

    grid = create_grid()
    sol_true = sol_klausmeier(hprm_true; t_fixed=t_fixed, t_end=t_end, t_step=t_step) # returns df
    sol_true = randomize_data!(sol_true, hprm_true.noise) # include noise

    ll = zeros(41, 21)

    for i in range(1, 41)
        for j in range(1, 21) #eval for each point on grid
            pt = grid[i,j]
            ll[i,j] = compute_ll([pt[1],pt[2]], hprm_true, sol_true; t_fixed=t_fixed, t_end=t_end, t_step=t_step)
        end
    end
    
    #return data frame
    a_eval_pts = string.(0.0:0.1:2.0)
    df_ll = DataFrame(ll, a_eval_pts)

    return df_ll
end

"""
    function gen_all_ll_data(index_combos::Vector{Vector{Int64}}, M_vals::Vector{Int64}, noise_vals::Vector{Float64}, m::Float64, w0::Float64, path::String; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)

function that generates and stores all the ll data needed. On all a,n0,M,noise prm combinations specifed.

# Arguments
- `index_combos::Vector{Vector{Int64}}`: indices of parameter values underlying true data observations.
- `M_val::Vector{Int64}`: sample sizes
- `noise_vals::Vector{Float64}`: noise levels
- `m::Float64`: mortality rate in Klausmeier model (fixed)
- `w0::Float64`: initial value for water compartment in Klausmeier model (fixed)
- `path::String`: path to folder where ll data is stored
- `t_fixed::Bool`: true if we consider a fixed observation time window
- `t_end::Float64`: end of observation window (if t_fixed=true)
- `t_step::Float64`: step size with which M observations should be picked (set if t_fixed=false)
"""
function gen_all_ll_data(index_combos::Vector{Vector{Int64}}, M_vals::Vector{Int64}, noise_vals::Vector{Float64}, m::Float64, w0::Float64, path::String; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)
    for ind in index_combos
        for M in M_vals
            for noise in noise_vals
                a_ind = ind[1]
                n0_ind = ind[2]
                hprm = Hyperprm(w0, n0_vals[n0_ind], a_vals[a_ind], m, M, noise)
                df_ll = gen_ll_evals_for_hprm_comb(hprm; t_fixed=t_fixed, t_end=t_end, t_step=t_step)
                store_ll_data(w0, n0_vals[n0_ind], a_vals[a_ind], m, M, noise, df_ll, path)
            end
        end
    end
end


# Functions for the fisher analysis
"""
    function store_fish_data(w0::Float64,m::Float64,M::Int64,noise::Float64,df::DataFrame, path::String)

stores data evaluated on grid in a csv file.
Name of form "fish_w0_n0_a_m_M_noise.csv"

# Arguments
- `df::DataFrame`: df to store
- `path_to_repo::String`: path to folder where to store the file
"""
function store_fish_data(w0::Float64,m::Float64,M::Int64,noise::Float64,df::DataFrame, path::String)
    CSV.write("$(path)fish_$(w0)_$(m)_$(M)_$(noise).csv", df)
end

"""
    function compute_fi(eval_pt::Vector{Float64}, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)

compute the Fisher information at evaluation point. The Fisher information is given by the trace of the negative Hessian of the log-likelihood function.

# Returns
- `Float64`: Fisher information value at given evaluation point
"""
function compute_fi(eval_pt::Vector{Float64}, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)
    H = ForwardDiff.hessian(x -> compute_ll(x, hprm, true_val; t_fixed=t_fixed, t_end=t_end, t_step=t_step), eval_pt)
    return tr(-H)
end

"""
    function gen_all_fish_data(M_vals, noise_vals, m, w0, path; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)

function that generates and stores all the fish data needed. On all a,n0,M,noise prm combinations specifed.

# Arguments
- `M_val::Vector{Int64}`: sample sizes
- `noise_vals::Vector{Float64}`: noise levels
- `m::Float64`: mortality rate in Klausmeier model (fixed)
- `w0::Float64`: initial value for water compartment in Klausmeier model (fixed)
- `path::String`: path to folder where fish data is stored
- `t_fixed::Bool`: true if we consider a fixed observation time window
- `t_end::Float64`: end of observation window (if t_fixed=true)
- `t_step::Float64`: step size with which M observations should be picked (set if t_fixed=false)
"""
function gen_all_fish_data(M_vals, noise_vals, m, w0, path; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)
    for M in M_vals
        for noise in noise_vals

            grid = create_grid()
            fish = zeros(41, 21)

            # keep track of whether the optimization algo terminates successfully when finding the MLE
            success_counter = 0
            eval_pt_counter = 0

            # evaluate fisher info on grid
            for i in range(1, 41)
                for j in range(1, 21)
                    eval_pt_counter = eval_pt_counter + 1 # total number of optimizations

                    pt = grid[i,j] # true observation parameter point
                    hprm = Hyperprm(w0, pt[2], pt[1], m, M, noise) # w0,n0,a,m,M

                    sol_true = sol_klausmeier(hprm; t_fixed=t_fixed, t_end=t_end, t_step=t_step)
                    sol_true = randomize_data!(sol_true, hprm.noise) # include noise

                    if noise == 0.0 # in this case stick to true prm combination
                        mle, success = [hprm.a, hprm.n0], true
                    else
                        mle, success = compute_mle(hprm, sol_true; t_fixed=t_fixed, t_end=t_end, t_step=t_step)
                    end

                    # evaluate Fi at MLE
                    fish[i,j] = compute_fi(mle, hprm, sol_true; t_fixed=t_fixed, t_end=t_end, t_step=t_step)

                    success_counter = success_counter + success # number of successfull optimizations
                end
            end

            success_fraction = success_counter / eval_pt_counter
            println("MLE terminated with success in $success_fraction cases.")
            
            # create data frame
            a_eval_pts = string.(0.0:0.1:2.0)
            df_fish = DataFrame(fish, a_eval_pts)

            store_fish_data(w0, m, M, noise, df_fish, path)
        end
    end
end