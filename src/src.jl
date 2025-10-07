module Src

    using DifferentialEquations, CSV, DataFrames, Plots, Distributions, ForwardDiff, LinearAlgebra, Random, Optim, DataStructures, GlobalSensitivityAnalysis, Distributions

    # parameter combinations for whose data simulations the log-likelihood can be analysed.
    a_vals = [0.1, 0.9, 1.1, 1.3, 1.7, 1.9, 0.8] 
    n0_vals = [0.2, 0.4, 1.0, 1.3, 2.3]


    include("model.jl") # code related to the klausmeier model
    include("data.jl") # code related to the compuatation of the log-likelihood and fisher information
    include("sobol.jl") # code related to gsa using sobol indices

end # module