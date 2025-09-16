export compute_sobol_indices

"""
    function compute_sobol_indices(N::Int64, dens_a::Distribution, dens_m::Distribution, dens_w0::Distribution, dens_n0::Distribution,
    M::Int64, t_fixed::Bool, t_end::Float64)

compute sobol indices using the package GlobalSensitivityAnalysis.jl

# Arguments
    - `N::Int64`: Sample size of sobol index computation
    - `dens_a::Distributiom`: density of a
    - `dens_m::Distribution`: density of m
    - `dens_w0::Distribution`: density of w0
    - `dens_n0::Distribution`: density of n0
    - `M::Int64`: number of measurements in model solution
    - `t_fixed::Bool`: indicates fixed time window in model simulation
    - `t_end::Float64`: indicates end of time window in model simulation

# Returns
    - `Matrix{Float64}`: matrix containting samples for Sobol index computation
"""
function compute_sobol_indices(N::Int64, dens_a::Distribution, dens_m::Distribution, dens_w0::Distribution, dens_n0::Distribution,
    M::Int64, t_fixed::Bool, t_end::Float64)

    # create Sobol data
    data = SobolData(
        params = OrderedDict(:a => dens_a,
        :m => dens_m,
        :w0 => dens_w0,
        :n0 => dens_n0),
        N=N
    )
    # create samples
    samples = GlobalSensitivityAnalysis.sample(data)

    # evaluate model on sample matrix
    n, w = zeros(size(samples, 1),M), zeros(size(samples, 1),M)
    for i in range(1:size(samples, 1)) 
        a, m, w0, n0 = samples[i, :]
        hprm = Src.Hyperprm(w0,n0,a,m,M,0.0) # no noise included
        sol = Src.sol_klausmeier(hprm, t_fixed=t_fixed, t_end=t_end)
        n[i,:], w[i,:] = sol[!, "n"], sol[!, "w"]
    end

    # compute sobol indices
    sobol_n = [analyze(data, n[:,j]) for j in 1:M]
    sobol_w = [analyze(data, w[:,j]) for j in 1:M]

    return sobol_n, sobol_w
end