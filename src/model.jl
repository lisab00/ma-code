"""Hyperprm <: AbstractHyperprm
All the parameters that can be set in the klausmeier model

# Fields:
* `w0`: init water compartment
* `n0`: init biomass compartment
* `a`: water input prm
* `m`: plant mortality prm
* `M`: sample size (number of measured time steps)
* `noise`: noise level of the data
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

"""define the klausmeier model equations
"""
function klausmeier!(du,u,p,t)
    du[1] = -u[1] - u[1] * u[2]^2 + p[1] # water compartment
    du[2] = u[1] * u[2]^2 - p[2] * u[2] # biomass compartment
end

"""solves the klausmeier model for given set of parameters
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