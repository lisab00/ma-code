"""
    Hyperprm <: AbstractHyperprm

All the parameters that can be set in the klausmeier model.

# Fields:
* `w0`: initial value for water compartment
* `n0`: initialvalue for biomass compartment
* `a`: water input model parameter
* `m`: plant mortality parameter
* `M`: sample size (number of measurements)
* `noise`: noise level (Gaussian variance) of the data
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
- `p`: p[1] denotes a (water input parameter), p[2] m (plant mortality rate) parameter
"""
function klausmeier!(du,u,p,t)
    du[1] = -u[1] - u[1] * u[2]^2 + p[1] # water compartment
    du[2] = u[1] * u[2]^2 - p[2] * u[2] # biomass compartment
end

"""
    function sol_klausmeier(hprm::Hyperprm; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)

solve/ simulate the klausmeier model for given set of parameters.
TODO: integration time?

# Arguments
- `hprm::Hyperprm`: parameters for which the Klausmeier simulation is performed
- `t_fixed::Bool`: true if we consider a fixed observation time window
- `t_end::Float64`: end of observation window (if t_fixed=true)
- `t_step::Float64`: TODO  // rm or fix

# Returns
- `DataFrame`: columns "t", "w", "n" represent the simulated state of the compartment at given time step
"""
function sol_klausmeier(hprm::Hyperprm; t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0)
    u0 = [hprm.w0; hprm.n0]
    p = [hprm.a; hprm.m]

    if t_fixed
        tspan = (0.0, t_end)
        prob = ODEProblem(klausmeier!, u0, tspan, p)
        steps = range(0.0, stop=t_end, length=hprm.M)
        sol = solve(prob,
            saveat=steps  # consider specific time points
        )
    else
        tspan = (0.0, hprm.M-1)
        prob = ODEProblem(klausmeier!, u0, tspan, p)
        sol = solve(prob,
            saveat=t_step  # consider specific time points
        )
    end
    return DataFrame(time=sol.t, w=sol[1, :], n=sol[2, :])
end