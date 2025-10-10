export compute_sobol_indices, sobol_index_subplot, sobol_index_subplot_wn

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
    for i in 1:size(samples, 1) 
        a, m, w0, n0 = samples[i, :]
        hprm = Hyperprm(w0,n0,a,m,M,0.0) # no noise included
        sol = sol_klausmeier(hprm, t_fixed=t_fixed, t_end=t_end)
        n[i,:], w[i,:] = sol[!, "n"], sol[!, "w"]
    end

    # compute sobol indices
    sobol_n = [analyze(data, n[:,j]) for j in 1:M]
    sobol_w = [analyze(data, w[:,j]) for j in 1:M]

    return sobol_n, sobol_w
end


# Plots
"""
    function sobol_index_subplot(sobol::Vector, comp::String; title::String="", M::Int64=1000)

create plot of sobol indices for all parameters for one trajectory.
    
# Arguments:
    - `sobol::Vector`: sobol object returned by index computation
    - `comp::String`: compartment to which indices belong
    - `title::String`: optional subplot title
"""
function sobol_index_subplot(sobol::Vector, comp::String; title::String="", M::Int64=1000, t_fixed::Bool=true, t_end=100.0)

    #colors = [:blue, :turquoise, :orange, :red]
    colors = [
        "#3070B3",  # a   → TUM blue brand
        "#F7811E",  # m   → TUM orange
        "#9ABCE4",  # w0  → TUM blue light-dark
        "#FAD080"   # n0  → TUM orange-2
    ]
    parameters = ["a", "m", "w0", "n0"];

    # extract indices for plotting
    fo = [sobol[i][:firstorder] for i in 1:M]
    to = [sobol[i][:totalorder] for i in 1:M]

    # very stupid way to create time axis
    hprm = Src.Hyperprm(1,1,1,1,M,0.0)
    sol = Src.sol_klausmeier(hprm, t_fixed=t_fixed, t_end=t_end)
    times = sol[!,"time"]

    si_plot = plot(legend=:topright)

    for k in 1:4
        col = colors[k]
        prm = parameters[k]
        plot!(times, getindex.(fo, k), label="$prm", lw=2, color=col, linestyle=:solid)
        plot!(times, getindex.(to, k), label="", lw=2, color=col, linestyle=:dash)
    end
    #xlabel!("Time")
    ylabel!("Sobol indices $comp")
    title!(title)
    return si_plot
end

"""
    function sobol_index_subplot_wn(sobol_n::Vector, sobol_w::Vector; title::String="", M::Int64=1000)

create plot of sobol indices for both trajectories underneath each other, for comparing.
    
# Arguments:
    - `sobol_n::Vector`: output of compute_sobol_indices for n compartment
    - `sobol_w::Vector`: output of compute_sobol_indices for w compartment
    - `title::String`: optional title
"""
function sobol_index_subplot_wn(sobol_n::Vector, sobol_w::Vector; title::String="", M::Int64=1000)

    # plot n compartment
    plot_n = sobol_index_subplot(sobol_n, "n", title = title, M=M)

    # plot w compartment
    plot_w = sobol_index_subplot(sobol_w, "w", M=M)

    # compare both
    return plot(plot_n, plot_w, layout=(2,1), size=(700,700))
end