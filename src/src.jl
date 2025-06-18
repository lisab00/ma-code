module Src

    using DifferentialEquations, CSV, DataFrames, Plots, Distributions, ForwardDiff, LinearAlgebra, Random

    a_vals = [0.1, 0.9, 1.1, 1.3, 1.7] 
    n0_vals = [0.2, 0.4, 1.0, 1.3, 2.3]


    include("model.jl")
    include("data.jl")

end # module