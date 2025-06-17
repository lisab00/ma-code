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
    w0::Float64
    n0::Float64
    a::Float64
    m::Float64
    M::Int64
    noise::Float64
end

"""define the klausmeier model equations
"""
function klausmeier!(du,u,p,t)
    du[1] = -u[1] - u[1] * u[2]^2 + p[1] # water compartment
    du[2] = u[1] * u[2]^2 - p[2] * u[2] # biomass compartment
end

"""solves the klausmeier model for given set of parameters
"""
function sol_klausmeier(hprm::Hyperprm)
    u0 = [hprm.w0; hprm.n0]
    p = [hprm.a; hprm.m]
    tspan = (0.0, hprm.M-1)

    prob = ODEProblem(klausmeier!, u0, tspan, p)
    sol = solve(prob,
        saveat=1.0  # consider specific time points
    )

    return DataFrame(time=sol.t, w=sol[1, :], n=sol[2, :])
end