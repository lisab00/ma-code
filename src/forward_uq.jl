export forward_uq

"""
    function forward_uq(mle::Vector, cov::Matrix, prm_keys::Vector, prm_true::Vector;
        t_pt_sample_dens::Int64=75, w0::Float64=1.0, n0::Float64=1.5, a::Float64=1.3, 
        m::Float64=0.45, M::Int64=100, n::Int64=100, t_fixed::Bool=false, 
        t_end::Float64=100.0, t_step::Float64=1.0)

Perform forward uncertainty quantification (UQ) using the Fisher approximation around the MLE.  
Samples parameter combinations from the Gaussian approximation `N(mle, cov)` and propagates them through 
the Klausmeier model to obtain probabilistic trajectories of the system states.

Also compares sampled trajectories to the true (noise-free) solution and visualizes:
- the probabilistic envelope of trajectories (`n` and `w`)
- the sample distributions of state values at a given observation time point.

# Arguments
    - `mle::Vector`: estimated MLE vector for parameters (mean of sampling distribution)
    - `cov::Matrix`: covariance matrix associated with MLE (used in Fisher approximation)
    - `prm_keys::Vector`: symbols of the parameters that are varied in the UQ (e.g., `[:a, :m]`)
    - `prm_true::Vector`: true underlying parameter values corresponding to `prm_keys`
    - `t_pt_sample_dens::Int64`: index of the time point at which to compute sample density plots
    - `n::Int64`: number of parameter samples to draw from the Fisher approximation

# Returns
    `NamedTuple` with fields:
    - `trajectories`: plot of probabilistic solution trajectories for `n` and `w`
    - `sample_dens_n`: plot of sampled density for `n` at `t_pt_sample_dens`
    - `sample_dens_w`: plot of sampled density for `w` at `t_pt_sample_dens`
"""
function forward_uq(mle::Vector, cov::Matrix, prm_keys::Vector, prm_true::Vector; t_pt_sample_dens::Int64=75,
    w0::Float64=1.0, n0::Float64=1.5, a::Float64=1.3, m::Float64=0.45, M::Int64=100, n::Int64=100
    , t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0)

    # compute sample trajectories
    n_traj_sampled, w_traj_sampled = sample_am_traj(mle, cov, prm_keys, n, w0=w0, n0=n0, a=a, m=m, M=M, t_fixed=t_fixed, t_end=t_end, t_step=t_step)

    # generate true data simulations
    prms = Dict(zip(prm_keys, prm_true))
    w0_val = get(prms, :w0, w0)
    n0_val = get(prms, :n0, n0)
    a_val  = get(prms, :a,  a)
    m_val  = get(prms, :m,  m)

    hprm = Hyperprm(w0_val, n0_val, a_val, m_val, M, 0.0)
    data_true = sol_klausmeier(hprm, t_fixed=t_fixed, t_end=t_end, t_step=t_step) # remove obs_late here because we want whole trajectory
    times = data_true[!,"time"]

    # plot probabilistic solution trajectories
    trajectories = plot_prob_traj(times, n_traj_sampled, w_traj_sampled, data_true)

    # plot sample densities at t_pt_sample_dens
    # true values at t_pt_sample_dens
    n_t_true, w_t_true = data_true[!,"n"][t_pt_sample_dens], data_true[!,"w"][t_pt_sample_dens]

    sample_dens_n = plot_sample_dens_t(t_pt_sample_dens, n_traj_sampled, n_t_true, "n")
    sample_dens_w = plot_sample_dens_t(t_pt_sample_dens, w_traj_sampled, w_t_true, "w")

    return (
        trajectories=trajectories,
        sample_dens_n=sample_dens_n,
        sample_dens_w=sample_dens_w,
    )
end

"""
    function sample_am_traj(mle::Vector, cov::Matrix, prm_keys::Vector, n::Int64;
        w0::Float64=1.0, n0::Float64=1.5, a::Float64=1.3, m::Float64=0.45, M::Int64=100,
        t_fixed::Bool=true, t_end::Float64=100.0, t_step::Float64=1.0)

Generate sample trajectories of the Klausmeier model based on the Fisher (Gaussian) approximation 
around the MLE. Samples `n` parameter combinations from the multivariate normal distribution 
`N(mle, cov)` and propagates them through the model to produce probabilistic solution ensembles.

# Arguments
    - `mle::Vector`: mean (MLE) of the Gaussian sampling distribution
    - `cov::Matrix`: covariance matrix (Fisher approximation of parameter uncertainty)
    - `prm_keys::Vector`: symbols of the parameters that are varied (e.g., `[:a, :m]`)
    - `n::Int64`: number of parameter samples to draw
    - `w0::Float64`: initial water concentration (default: 1.0)
    - `n0::Float64`: initial nutrient concentration (default: 1.5)
    - `a::Float64`: growth rate parameter (default: 1.3)
    - `m::Float64`: mortality rate parameter (default: 0.45)
    - `M::Int64`: number of observation points in the simulation
    - `noise::Float64`: noise level added to simulated data (default: 0.0)
    - `t_fixed::Bool`: whether to simulate over a fixed time window (`true`) or variable (`false`)
    - `t_end::Float64`: end of simulation time window (if `t_fixed=true`)
    - `t_step::Float64`: time step between observations (if `t_fixed=false`)

# Returns
    `Tuple{Vector, Vector}`:
    - `n_traj_sampled`: vector of sampled nutrient trajectories
    - `w_traj_sampled`: vector of sampled water trajectories

Each trajectory corresponds to one sampled parameter realization drawn from the Fisher approximation.
"""
function sample_am_traj(mle::Vector, cov::Matrix, prm_keys::Vector, n::Int64; w0::Float64=1.0, n0::Float64=1.5, a::Float64=1.3, m::Float64=0.45, M::Int64=100, t_fixed::Bool=true, t_end::Float64=100.0, t_step::Float64=1.0)
    # Fisher approximation
    if length(mle) == 1
        dist = Normal(mle[1], sqrt(cov[1]))
        samples = rand(dist, n)'
    else
        cov = Symmetric((cov + cov') / 2) # ensure numerical stability
        dist = MvNormal(mle, cov)
        samples = rand(dist, n)
    end
    
    # store for plotting
    n_traj_sampled = []
    w_traj_sampled = []

    # for each sample solve klausmeier model
    for i in 1:size(samples,2)
        s = samples[:,i]

        prms = Dict(zip(prm_keys, s))
        w0_val = get(prms, :w0, w0)
        n0_val = get(prms, :n0, n0)
        a_val  = get(prms, :a,  a)
        m_val  = get(prms, :m,  m)

        # Build hyperparameter object
        hprm = Hyperprm(w0_val, n0_val, a_val, m_val, M, 0.0)

        sol = sol_klausmeier(hprm, t_fixed=t_fixed, t_end=t_end,t_step=t_step) # leave out obs_late here because we want to simulate trajectory from beginning
        push!(n_traj_sampled, sol[!,"n"])
        push!(w_traj_sampled, sol[!,"w"])
    end
    return n_traj_sampled, w_traj_sampled
end

"""
    function plot_prob_traj(times::Vector{Float64}, n_traj_sampled::Vector{Any}, w_traj_sampled::Vector{Any}, data_true::DataFrame)

Plot probabilistic trajectories of model simulations. Black line indicates true, noiseless solution.

# Inputs:
    - `times::Vector{Float64}"`: considered time points of solution
    - `n_traj_sampled::Vector{Any}`: each item is vector of one sampled trajectory of n
    - `w_traj_sampled::Vector{Any}`: each item is vector of one sampled trajectory of w
    - `data_true::DataFrame`: output of sol_klausmeier for true parameter values (not noisy!)
"""
function plot_prob_traj(times::Vector{Float64}, n_traj_sampled::Vector{Any}, w_traj_sampled::Vector{Any}, data_true::DataFrame)
    n_mean = mean(n_traj_sampled, dims=1)
    w_mean = mean(w_traj_sampled, dims=1)

    n = length(n_traj_sampled)

    plot_traj = plot(times, n_mean, label="mean n", lw=2, color="#F7811E", legend=:bottomright, title="")
    plot!(times, w_mean, label="mean w", lw=2, color="#3070B3")
    for i in range(1, n)
        plot!(times,n_traj_sampled[i], color="#F7811E", alpha=0.05, label="")
        plot!(times,w_traj_sampled[i], color="#3070B3", alpha=0.05,label="")
    end
    plot!(times, data_true[!,"n"], lw=2, color=:black, label="",linestyle=:dash)
    plot!(times, data_true[!,"w"], lw=2, color=:black, label="", linestyle=:dash)
    return plot_traj
end

"""
    function plot_sample_dens_t(t_pt_sample_dens::Int64, traj_sampled::Vector{Any}, traj_t_true::Float64, traj_name::String)

Plots the sample density of the trajectories at specified time point. Vertical lines indicate true values of noiseless solution at t_pt_sample_dens.

# Args:
    - `t_pt_sample_dens::Int64`: time point at which sample density is computed
    - `traj_sampled::Vector{Any}`: each item is vector of one sampled trajectory
    - `traj_t_true::Float64`: true value at time point of trajectory (not noisy!)
    - `traj_name::String`: name of the trajectory considered in samples
"""        
function plot_sample_dens_t(t_pt_sample_dens::Int64, traj_sampled::Vector{Any}, traj_t_true::Float64, traj_name::String)
    traj_t_sam = [s[t_pt_sample_dens] for s in traj_sampled]
    traj_t_dens = kde(traj_t_sam)
    plot_dens = plot(traj_t_dens.x, traj_t_dens.density, color="#3070B3", lw=2, ylabel="sampled pdf", label="$traj_name", title="")
    vline!([traj_t_true], color=:black, linestyle=:dash, label="true value")
    return plot_dens
end


