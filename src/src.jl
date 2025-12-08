module Src

    using DifferentialEquations, ForwardDiff, Optim, GlobalSensitivityAnalysis 
    using CSV, DataFrames, Plots, DataStructures, LinearAlgebra, LaTeXStrings, Plots.Measures
    using Random, Distributions, KernelDensity


    # custom color gradient for plots
    tum_blues = ["#C2D7EF", "#9ABCE4", "#5E94D4", "#165DB1"]
    tum_cgrad = cgrad(tum_blues)


    include("model.jl") # code related to the klausmeier model
    include("data.jl") # code related to the computation of the log-likelihood and fisher information
    include("sobol.jl") # code related to global sensitivity analysis using sobol indices
    include("identifiability.jl") # code related to the inverse problem
    include("forward_uq.jl") # code related to forward UQ
    include("plots_ll_fish.jl") # code related to plotting the ll and fisher surfaces

end # module