export Hyperprm, sol_klausmeier, randomize_data!, bif_plot, bif_plot!

"""
    Hyperprm <: AbstractHyperprm

All the parameters that can be set in the klausmeier model.

# Fields:
    - `w0`: initial value for water compartment
    - `n0`: initialvalue for biomass compartment
    - `a`: water input model parameter
    - `m`: plant mortality parameter
    - `M`: sample size (number of measurements)
    - `noise`: noise level (Gaussian variance) of the data
"""
abstract type AbstractHyperprm end

struct Hyperprm <: AbstractHyperprm
    w0::Real
    n0::Real
    a::Real
    m::Real
    M::Real
    noise::Real
end

"""
    function klausmeier!(du,u,p,t)

define the klausmeier model equations.

# Arguments
    - `du, u`: u[1], du[1] denote water compartment w. u[2], du[2] biomass compartment n
    - `p`: p[1] denotes "a" (water input parameter), p[2] "m" (plant mortality rate) parameter
"""
function klausmeier!(du,u,p,t)
    du[1] = -u[1] - u[1] * u[2]^2 + p[1] # water compartment
    du[2] = u[1] * u[2]^2 - p[2] * u[2] # biomass compartment
end

"""
    function sol_klausmeier(hprm::Hyperprm; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)

solve/ simulate the klausmeier model for given set of parameters and select number of observations samples M.
Integration time window can either be fixed (to t_end) or variable.
In the former case M denotes the sample density within fixed time window. In the latter case integration time ends after observing M samples of time distance t_step.
The model is always solved with mesh size 0.1 and M samples are taken equidistantly.
Optionally, only late observations starting at `t_obs` can be returned by setting `obs_late=true`.

# Arguments
    - `hprm::Hyperprm`: parameters for which the Klausmeier simulation is performed
    - `t_fixed::Bool`: true if we consider a fixed observation time window
    - `t_end::Float64`: end of observation window (set if t_fixed=true)
    - `t_step::Float64`: step size with which M observations should be picked (set if t_fixed=false)
    - `obs_late::Bool=false`: If true, only return observations starting at time `t_obs`
    - `t_obs::Float64=100.0`: Time at which late observations begin (used if `obs_late=true`)

# Returns
    - `DataFrame`: columns "time", "w", "n" represent the simulated state of the compartment at given time step
"""
function sol_klausmeier(hprm::Hyperprm; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)
    u0 = [hprm.w0; hprm.n0]
    p = [hprm.a; hprm.m]

    if t_fixed
        tspan = (0.0, t_end)
    else
        M_end =  t_step*(hprm.M-1)
        tspan = (0.0, M_end) # ensure we sample up to sufficient time (later pick M samples with step size t_step)

    end

    prob = ODEProblem(klausmeier!, u0, tspan, p)
    sol = solve(prob,
        saveat=0.1  # FE step size 0.1, save at equidistant range
    )
    df_sol = DataFrame(time=sol.t, w=sol[1, :], n=sol[2, :])

    if t_fixed
        df_sol = select_M_rows(df_sol, hprm.M) # pick M equidistant samples from fixed observation time window
    else
        df_sol = step_M_times(df_sol, M_end, t_step) # pick M consecutive time steps with step size t_step
    end

    if obs_late # only return observations starting at time step obs_ind
        df_sol = df_sol[df_sol.time .>= t_obs, :]
    end
    
    return df_sol
end

"""
    function select_M_rows(df::DataFrame, M::Real)

select M rows in equidistant steps from DataFrame.

# Arguments
    - `df::DataFrame`: DataFrame from which rows are selected
    - `M::Real`: Number of rows to select

# Returns
    - `DataFrame`: contains M rows=samples of ODE solution DataFrame
"""
function select_M_rows(df::DataFrame, M::Real)
    indices = round.(Int, range(1, nrow(df), length=M))
    return df[indices, :]
end

"""
    function step_M_times(df::DataFrame, M_end::Float64, t_step::Float64)

start at t=0 and make time steps with length t_step until M_end. M_end=M*t_step such that we obtain M equidistand time steps.

# Arguments
    - `df::DataFrame`: contain ODE solution with "time" column
    - `M_end::Float64`: end time point of ODE observation
    - `t_step::Float64`: step size of times at which we want to pick the observations

# Returns
    - `DataFrame`: contains M samples observed in time distances t_step
"""
function step_M_times(df::DataFrame, M_end::Float64, t_step::Float64)
    time_pts = 0:t_step:M_end 
    df = filter(row -> row.time in time_pts, df)
    return df
end


### Bifurcation Plots
"""
    function bif_plot(m::Float64, comp::String; points=[])

Plot the bifurcation diagram of the Klausmeier model for a given mortality rate `m`.  
The bifurcation curves are computed for either the biomass (`n`) or water (`w`) steady-state 
components as a function of the water input parameter `a`.

# Arguments
    - `m::Float64`: mortality rate parameter in the Klausmeier model
    - `comp::String`: specifies which component to plot — `"n"` for biomass or `"w"` for water
    - `points=[]`: optional list of `(a, value)` tuples to highlight specific points on the bifurcation diagram
"""
function bif_plot(m::Float64, comp::String; points=[])
    a_vals = range(2*m, stop=2, length=400)
    
    if comp == "n"
        plus = [n(a, m, true) for a in a_vals]
        minus = [n(a, m, false) for a in a_vals]
    else
        plus = [w(a, m, true) for a in a_vals]
        minus = [w(a, m, false) for a in a_vals]
    end

    # Plot the bifurcation curves
    plot(a_vals, plus, color=:black, label="", lw=3, xlim=(0, maximum(a_vals)), ylim=(0, 2))
    plot!(a_vals, minus, color=:black, linestyle=:dash, label="", lw=3)
    
    # Plot points if provided
    for pt in points
        scatter!([pt[1]], [pt[2]], color="#3070B3", marker=:x, markersize=8, markerstrokewidth=6, label="")
    end

    # Axes labels, title
    xlabel!("water input a")
    if comp == "n"
        scatter!([2*m], [1], color=:black, marker=:circle, markersize=6, label="")
        hline!([0], color=:black,  lw=5, linestyle=:solid, label="")
        ylabel!("biomass n")
    else
        scatter!([2*m], [m], color=:black, marker=:circle, markersize=6, label="")
        plot!(0:0.1:2, 0:0.1:2, color=:black,  lw=3, linestyle=:solid, label="")
        ylabel!("water input w")
    end
    #title!("Bifurcation Diagram of Klausmeier Model\nm = $m")
end

"""
    function bif_plot!(m::Float64, comp::String)

adds bifurcation diagram to an existing plot (for documentation see bif_plot)
"""
function bif_plot!(m::Float64, comp::String)
    a_vals = range(2*m, stop=2, length=400)
    
    if comp == "n"
        plus = [n(a, m, true) for a in a_vals]
        minus = [n(a, m, false) for a in a_vals]
    else
        plus = [w(a, m, true) for a in a_vals]
        minus = [w(a, m, false) for a in a_vals]
    end

    # Plot the bifurcation curves
    plot!(a_vals, plus, color=:black, label="", lw=3, xlim=(0, maximum(a_vals)), ylim=(0, 2))
    plot!(a_vals, minus, color=:black, linestyle=:dash, label="", lw=3)
    
    # Axes labels, title
    xlabel!("water input a")
    if comp == "n"
        scatter!([2*m], [1], color=:black, marker=:circle, markersize=6, label="")
        hline!([0], color=:black,  lw=5, linestyle=:solid, label="")
        ylabel!("biomass n")
    else
        scatter!([2*m], [m], color=:black, marker=:circle, markersize=6, label="")
        plot!(0:0.1:2, 0:0.1:2, color=:black,  lw=3, linestyle=:solid, label="")
        ylabel!("water w")
    end
    #title!("Bifurcation Diagram of Klausmeier Model\nm = $m")
end


# biomass equilibrium branches
"""
    n(a, m, plus::Bool)

Compute the biomass equilibrium branch of the Klausmeier model.

# Arguments
    - `a::Float64`: water input parameter.
    - `m::Float64`: mortality rate.
    - `plus::Bool`: if `true`, return the upper (stable) branch; if `false`, return the lower (unstable) branch.

# Returns
    - `Float64`: equilibrium biomass value.
"""
function n(a, m, plus::Bool)
    sqrt_term = sqrt(a^2 - 4*m^2)
    if plus
        return (a + sqrt_term) / (2*m)
    else
        return (a - sqrt_term) / (2*m)
    end
end

# water equilibrium branches
"""
    w(a, m, plus::Bool)

Compute the water equilibrium branch of the Klausmeier model.

# Arguments
    - `a::Float64`: water input parameter.
    - `m::Float64`: mortality rate.
    - `plus::Bool`: if `true`, return the upper (stable) branch; if `false`, return the lower (unstable) branch.

# Returns
    - `Float64`: equilibrium water value.
"""
function w(a, m, plus::Bool)
    sqrt_term = sqrt(a^2 - 4*m^2)
    if plus
        return (2*m^2) / (a + sqrt_term)
    else
        return (2*m^2) / (a - sqrt_term)
    end
end
