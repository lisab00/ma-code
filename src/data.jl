export gen_store_ll_data, gen_all_fish_data_an0_plane


# Tools
"""
    function create_grid()

Create the evaluation grid for a function. We evaluate for x,y in (0,2) on a uniformly spaced grid with mesh size 0.01
"""
function create_grid()
    x_vals = 0.0:0.01:2.0
    y_vals = 0.0:0.01:2.0

    grid = [(x, y) for y in y_vals, x in x_vals] 
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
    function compute_mle(prm_keys::Vector, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0, N::Int64=5)

compute the maximum likelihood estimate given data observations by minimizing the negative log-likelihood function using the Optim.jl package.
We employ multiple restart optimization to obtain a robust estimate.

# Arguments
    - `prm_keys::Vector`: Names of the parameters to be estimated (e.g., `[:a, :m]`)
    - `hprm::Hyperprm`: Hyperparameter struct defining the true underlying model and noise characteristics
    - `true_val::DataFrame`: Observed data used for likelihood computation
    - `t_fixed::Bool=false`: True if fixed observation time window is considered
    - `t_end::Float64=100.0`: End of observation window (used if `t_fixed=true`)
    - `t_step::Float64=1.0`: Time step for observation sampling (used if `t_fixed=false`)
    - `obs_late::Bool=false`: If true, only use observations starting at time `t_obs`
    - `t_obs::Float64=100.0`: Time at which late observations begin (used if `obs_late=true`)
    - `N::Int64=5`: Number of random restarts in the optimization procedure

# Returns
    - `Vector{Float64}`: vector containing the MLE (e.g., `[a_mle, m_mle]`)
    - `Bool`: `true` if the optimization converged successfully
"""
function compute_mle(prm_keys::Vector, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0, N::Int64=5)
    inits, inits_loss, mles, losses, best_loss_ind, converged = mult_restart_mle(N, prm_keys, hprm, true_val; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)
    return mles[best_loss_ind, :], converged[best_loss_ind]
end

"""
    function mult_restart_mle(N::Int64, prm_keys::Vector, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)
 
Perform Maximum Likelihood estimation for N different starting points. Goal is to find global minimum.

# Arguments
    - `N::Int64`: Number of random restart trials
    - `prm_keys::Vector`: Names of parameters to be estimated (e.g., `[:a, :n0]`)
    - `hprm::Hyperprm`: Hyperparameter struct defining model configuration and noise
    - `true_val::DataFrame`: Observed data used for likelihood evaluation

# Returns
    - `Matrix`: initial values used in optimization
    - `Vector`: losses of initial values
    - `Matrix`: computed MLEs
    - `Vector`: corresponding losses of MLEs
    - `Int`: index of optimization trial creating minimal loss
    - `Vector`: convergence status for each optimization trial
"""
function mult_restart_mle(N::Int64, prm_keys::Vector, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)
    
    # number of parameters to optimize
    n_prms = length(prm_keys)   

    # generate random optimization start pts
    inits = hcat([2 .* rand(N) for _ in 1:n_prms]...)  

    # store mles and corresponding loss
    mle_vals = zeros(N, n_prms)
    mle_loss, inits_loss, converged = zeros(N), zeros(N), zeros(N)

    for i in 1:N
        pt = inits[i,:]
        result = optimize(x -> - compute_ll(x, prm_keys, hprm, true_val; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs), pt)
        #display(result)
        mle_vals[i,:] = Optim.minimizer(result)
        mle_loss[i] =  Optim.minimum(result)
        converged[i] = Optim.converged(result)
        inits_loss[i] = -compute_ll(pt, prm_keys, hprm, true_val; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)
    end

    # extract best
    best_loss, best_loss_ind = findmin(mle_loss)

    return inits, inits_loss, mle_vals, mle_loss, best_loss_ind, converged
end


# Functions for the likelihood analysis
"""
    function store_ll_data(w0::Float64,n0::Float64,a::Float64,m::Float64,M::Int64,noise::Float64,df::DataFrame, path_to_store::String)

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
    function compute_ll(x::Vector, prm_keys::Vector, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)

compute the log-likelihood in least-squares form for Klausmeier model for data with Gaussian noise. First, simulate Klausmeier model for given hyperparameters and noise level. Then, compare to true trajectories.
Includes x variables needed for ForwardDiff and Optim.

# Arguments
    - `x::Vector`: Parameter values (in order specified by `prm_keys`) used in the likelihood computation
    - `prm_keys::Vector`: Names of parameters to be estimated (e.g., `[:a, :m]`)
    - `hprm::Hyperprm`: Hyperparameter struct defining model constants and noise level
    - `true_val::DataFrame`: Observed trajectories with columns `"w"` and `"n"`

# Returns
    - `Float`: scalar value of log-likelihood at given grid point 
"""
function compute_ll(x::Vector, prm_keys::Vector, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)

    # determine which parameters are of interest
    prms = Dict(zip(prm_keys, x)) # returnd dict with parameter name and value given by x

    # update respective parameters
    # if prms contains entry then update, if not take previous value
    a     = get(prms, :a, hprm.a)
    n0    = get(prms, :n0, hprm.n0)
    m     = get(prms, :m, hprm.m)
    w0     = get(prms, :w0, hprm.w0)

    hprm = Hyperprm(w0, n0, a, m, hprm.M, hprm.noise)
    pred_val = sol_klausmeier(hprm; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs) # this is not noisy!
    
    # assume noisy of true data to be known
    if hprm.noise == 0.0
        ll = -0.5 * sum((true_val[:,"n"] - pred_val[:,"n"]) .^2) - 0.5 * sum((true_val[:,"w"] - pred_val[:,"w"]) .^2) # add up ll for both trajectories
    else
        ll = -0.5 * 1/hprm.noise * sum((true_val[:,"n"] - pred_val[:,"n"]) .^2) - 0.5 * 1/hprm.noise * sum((true_val[:,"w"] - pred_val[:,"w"]) .^2) # add up ll for both trajectories
    end
    return ll
end

"""
    function gen_ll_evals(prm_keys::Vector, hprm_true::Hyperprm; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)
 
Evaluates the log-likelihood evaluations of parameters for visualization or identifiability analysis.

# Arguments
    - `prm_keys::Vector`: Names of the two parameters to evaluate on the grid
    - `hprm_true::Hyperprm`: True hyperparameter values used to simulate data
    - `t_fixed::Bool=false`: True if a fixed observation time window is considered
    - `t_end::Float64=50.0`: End of the observation window (if t_fixed=true)
    - `t_step::Float64=1.0`: Step size for observations (if t_fixed=false)
    - `obs_late::Bool=false`: True if only late (stable state) observations are considered
    - `t_obs::Float64=100.0`: Time at which late observations are taken if `obs_late=true`

# Returns
    - A `DataFrame` containing the log-likelihood values over the 2D parameter grid
"""
function gen_ll_evals(prm_keys::Vector, hprm_true::Hyperprm; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)
    
    # create true data observations
    sol_true = sol_klausmeier(hprm_true; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs) # returns df
    sol_true = randomize_data!(sol_true, hprm_true.noise)

    if length(prm_keys)==2
        grid = create_grid()
        ll = zeros(size(grid, 1), size(grid, 2))
        for i in range(1, size(grid, 1))
            for j in range(1, size(grid, 2)) #eval for each point on grid
                pt = grid[i,j]
                ll[i,j] = compute_ll([pt[1],pt[2]], prm_keys, hprm_true, sol_true; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)
            end
        end
        return DataFrame(ll,:auto)
    else
        xr = 0.0:0.01:2.0
        ll = zeros(length(xr))
        for i in range(1, length(xr))
            pt = xr[i]
            ll[i] = compute_ll([pt[1]], prm_keys, hprm_true, sol_true; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)
        end
        return DataFrame(ll=ll)
    end
end

"""
    function gen_store_ll_data(points::Vector{Vector{Float64}}, prm_keys::Vector, M_vals::Vector{Int64}, noise_vals::Vector{Float64}, path::String; a::Float64=1.3, m::Float64=0.45, n0::Float64=1.0, w0::Float64=1.0, t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)

function that generates and stores all the ll data needed. On all a,n0,M,noise prm combinations specifed.

# Arguments
    - `points::Vector{Vector{Float64}}`: Parameter grid points (e.g. combinations of `[a, m]`) for which LL data are computed
    - `prm_keys::Vector`: Names of parameters corresponding to each entry in `points`
    - `M_vals::Vector{Int64}`: Sample sizes used in simulations
    - `noise_vals::Vector{Float64}`: Noise levels
    - `path::String`: Path to the folder where ll data will be stored

# Keyword Arguments
    - `a::Float64=1.3`: Default water input rate in the Klausmeier model (used if not included in `prm_keys`)
    - `m::Float64=0.45`: Mortality rate (fixed unless included in `prm_keys`)
    - `n0::Float64=1.0`: Initial nutrient concentration (fixed unless included in `prm_keys`)
    - `w0::Float64=1.0`: Initial water concentration (fixed unless included in `prm_keys`)
    - `t_fixed::Bool=false`: If true, simulate over a fixed time window `[0, t_end]`
    - `t_end::Float64=100.0`: End time of observation window (if `t_fixed=true`)
    - `t_step::Float64=1.0`: Time step between observations (if `t_fixed=false`)
    - `obs_late::Bool=false`: If true, use only observations taken in the steady-state regime
    - `t_obs::Float64=100.0`: Time at which late observations are taken (if `obs_late=true`)

# Description
For each combination of `(a, n0, M, noise)`, this function:
    1. Builds the corresponding `Hyperprm` object.
    2. Computes log-likelihood evaluations using `gen_ll_evals`.
    3. Saves the resulting data to the specified directory using `store_ll_data`.
"""
function gen_store_ll_data(points::Vector{Vector{Float64}}, prm_keys::Vector, M_vals::Vector{Int64}, noise_vals::Vector{Float64}, path::String; a::Float64=1.3, m::Float64=0.45, n0::Float64=1.0, w0::Float64=1.0, t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)
    for pt in points
        for M in M_vals
            for noise in noise_vals
                prms = Dict(zip(prm_keys, pt))
                a_val = get(prms, :a, a)
                n0_val = get(prms, :n0, n0)
                m_val = get(prms, :m, m)
                w0_val = get(prms, :w0, w0)
                hprm = Hyperprm(w0_val, n0_val, a_val, m_val, M, noise)
                df_ll = gen_ll_evals(prm_keys, hprm; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)
                store_ll_data(w0_val, n0_val, a_val, m_val, M, noise, df_ll, path)
            end
        end
    end
end


# Functions for the fisher analysis
"""
    function store_fish_data(M::Int64,noise::Float64,df::DataFrame, path::String)

stores data evaluated on grid in a csv file.
Name of form "fish_M_noise.csv"

# Arguments
    - `df::DataFrame`: df to store
    - `path_to_repo::String`: path to folder where to store the file
"""
function store_fish_data(M::Int64,noise::Float64,df::DataFrame, path::String)
    CSV.write("$(path)fish_$(M)_$(noise).csv", df)
end

"""
    function compute_fi(eval_pt::Vector{Float64}, prm_keys::Vector, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)

compute the Fisher information at evaluation point. The Fisher information is given by the trace of the negative Hessian of the log-likelihood function.

# Arguments
    - `eval_pt::Vector{Float64}`: Parameter vector at which FI is evaluated
    - `prm_keys::Vector`: Names of parameters corresponding to entries in `eval_pt`
    - `hprm::Hyperprm`: Model hyperparameters used for simulation and likelihood computation
    - `true_val::DataFrame`: Observed (noisy) trajectories with columns `"w"` and `"n"`

# Returns
    - `Float64`: Fisher information value at given evaluation point
"""
function compute_fi(eval_pt::Vector{Float64}, prm_keys::Vector, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)
    H = ForwardDiff.hessian(x -> compute_ll(x, prm_keys, hprm, true_val; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs), eval_pt)
    return tr(-H)
end

"""
    function gen_all_fish_data_prm_plane(prm_keys::Vector, M_vals::Vector, noise_vals::Vector, path::String; a::Float64=1.3, m::Float64=0.45, n0::Float64=1.0, w0::Float64=1.0, t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)
"""
# brauch ich das? ggf noch erweitern auf 1D parameter
function gen_all_fish_data_prm_plane(prm_keys::Vector, M_vals::Vector, noise_vals::Vector, path::String; a::Float64=1.3, m::Float64=0.45, n0::Float64=1.0, w0::Float64=1.0, t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)
    for M in M_vals
        for noise in noise_vals

            grid = create_grid()
            fish = zeros(201, 201)

            # keep track of whether the optimization algo terminates successfully when finding the MLE
            success_counter = 0
            eval_pt_counter = 0

            # evaluate fisher info on grid
            for i in range(1, 201)
                for j in range(1, 201)
                    eval_pt_counter = eval_pt_counter + 1 # total number of optimizations

                    pt = grid[i,j] # true observation parameter point

                    # assign point values to prm_keys
                    prms = Dict(zip(prm_keys, pt))
                    a_val  = get(prms, :a, a)
                    n0_val = get(prms, :n0, n0)
                    m_val  = get(prms, :m, m)
                    w0_val = get(prms, :w0, w0)

                    hprm = Hyperprm(w0_val, n0_val, a_val, m_val, M, noise)

                    sol_true = sol_klausmeier(hprm; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)
                    sol_true = randomize_data!(sol_true, hprm.noise) # include noise

                    mle, success = compute_mle(prm_keys, hprm, sol_true; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)

                    # evaluate Fi at MLE
                    fish[i,j] = compute_fi(mle, prm_keys, hprm, sol_true; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)

                    success_counter = success_counter + success # number of successfull optimizations
                end
            end

            success_fraction = success_counter / eval_pt_counter
            println("MLE terminated with success in $success_fraction cases.")
            
            # create data frame
            x_eval_pts = string.(0.0:0.01:2.0)
            df_fish = DataFrame(fish, x_eval_pts)

            store_fish_data(M, noise, df_fish, path)
        end
    end
end

"""
    function gen_all_fish_data_an0_plane(prm_keys::Vector, M_vals::Vector, noise_vals::Vector, path::String;
                                         m::Float64=0.45, w0::Float64=1.0, t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0, N::Int64=5)

Generate and store Fisher Information data across a grid of `(a, n₀)` parameter values for different sample sizes and noise levels.  
For each parameter combination, the Klausmeier model is simulated, the MLE is estimated via multiple restarts, and the Fisher Information is computed at the MLE.

# Arguments
    - `prm_keys::Vector`: Names of parameters to be estimated (e.g., `[:a, :m]`)
    - `M_vals::Vector`: Sample sizes (number of observations per trajectory)
    - `noise_vals::Vector`: Noise levels applied to simulated data
    - `path::String`: Directory path where FI data will be stored
    - `m::Float64=0.45`: Mortality rate in the Klausmeier model (fixed)
    - `w0::Float64=1.0`: Initial water compartment value (fixed)
    - `t_fixed::Bool=false`: If true, integrate over a fixed observation window `[0, t_end]`
    - `t_end::Float64=100.0`: End time of observation window (if `t_fixed=true`)
    - `t_step::Float64=1.0`: Step size between observations (if `t_fixed=false`)
    - `obs_late::Bool=false`: If true, only consider late-time observations
    - `t_obs::Float64=100.0`: Time at which late observations start (if `obs_late=true`)
    - `N::Int64=5`: Number of optimization restarts used for MLE estimation

# Behavior
    - Loops over all combinations of `M_vals` and `noise_vals`
    - Constructs a parameter grid in `(a, n₀)` space
    - Simulates data with Gaussian noise for each grid point
    - Computes the MLE using multiple-restart optimization
    - Evaluates Fisher Information at the MLE
    - Stores the resulting FI matrix as a CSV file via `store_fish_data`
"""
function gen_all_fish_data_an0_plane(prm_keys::Vector, M_vals::Vector, noise_vals::Vector, path::String; m::Float64=0.45, w0::Float64=1.0, t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0, N::Int64=5)
    for M in M_vals
        for noise in noise_vals

            grid = create_grid()
            fish = zeros(201, 201)

            # keep track of whether the optimization algo terminates successfully when finding the MLE
            success_counter = 0
            eval_pt_counter = 0

            # evaluate fisher info on grid
            for i in range(1, 201)
                for j in range(1, 201)
                    eval_pt_counter = eval_pt_counter + 1 # total number of optimizations

                    pt = grid[i,j] # true observation parameter point combination of a, n0
                    hprm = Hyperprm(w0, pt[2], pt[1], m, M, noise) # w0,n0,a,m,M

                    sol_true = sol_klausmeier(hprm; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)
                    sol_true = randomize_data!(sol_true, hprm.noise) # include noise

                    mle, success = compute_mle(prm_keys, hprm, sol_true; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs, N=N)

                    # evaluate Fi at MLE
                    fish[i,j] = compute_fi(mle, prm_keys, hprm, sol_true; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)

                    success_counter = success_counter + success # number of successfull optimizations
                end
            end

            success_fraction = success_counter / eval_pt_counter
            println("MLE terminated with success in $success_fraction cases.")
            
            # create data frame
            a_eval_pts = string.(0.0:0.01:2.0)
            df_fish = DataFrame(fish, a_eval_pts)

            store_fish_data(M, noise, df_fish, path)
        end
    end
end