# ma-code
This repository complements the work done in my master's thesis 

    "The Role of Bifurcations in Parameter Estimation: Application to the Klausmeier Vegetation Model with Random Coefficients".

The code implements uncertainty quantification (UQ) and parameter estimation tools from [2] and applies them to the Klausmeier vegetation model in its ODE form [1].
The goal is to systematically assess parameter identifiability and uncertainty around the model bifurcation, expanding the work of [3].
We treat the model in a Bayesian setting and try to estimate posterior parameter distributions using Gaussian approximation of the posterior.

The following are implemented (amongst others):
- Global sensitvity analysis using Sobol indices
- Solution and UQ assessment of the inverse problem 
- Forward UQ with Monte Carlo simulations
- Simulation of model trajectories and bifurcation behavior

In `src/` all the functionalities are implemented; `notebooks/` contains all experiments conducted in Jupyter notebooks; `plots/` stores all resulting plots.

##### Main references:
<sub>
[1] <b>Klausmeier, C. A.</b> (1999). "Regular and irregular patterns in semiarid vegetation." <i>Science</i>, 284(5421), pp. 1826-1828.<br>
[2] <b>Piazzola, C., et al.</b> (2021). "A note on tools for prediction under uncertainty and identifiability of SIR-like dynamical systems for epidemiology." In: Mathematical Biosciences 332 (2021), p. 108514.<br>
[3] <b>Roesch, E., & Stumpf, M. P. H.</b> (2019). "Parameter inference in dynamical systems with co-dimension 1 bifurcations." <i>Royal Society Open Science</i>, 6(10).
</sub>
