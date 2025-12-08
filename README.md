# ma-code
This repository complements the work done in my master's thesis "The Role of Bifurcations in Parameter Estimation: Application to the Klausmeier Vegetation Model with Random Coefficients".
It implements uncertainty quantification and parameter estimation tools from [2] and applies them to the Klausmeier vegetation model in its ODE form [1].
The goal is to systematically assess parameter identifiability and uncertainty around the model bifurcation, expanding the work of [3].








Main references:
- [1] **Klausmeier, C. A.** (1999). "Regular and irregular patterns in semiarid vegetation." *Science*, 284(5421), 1826-1828.
    * *Source of the biological model used in this study.*
- [2] **Piazzola, C., et al.** (2021)
    * *Note: Referenced as the foundational workflow for the uncertainty quantification analysis performed in this work.*
- [3] **Roesch, E., & Stumpf, M. P. H.** (2019). "Parameter inference in dynamical systems with co-dimension 1 bifurcations." *Royal Society Open Science*, 6(10).
    * *Methodological basis for using Fisher Information to assess identifiability near bifurcation points.*
