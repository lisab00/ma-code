export assess_practical_identifiability, analyze_ll

# implements whole routine, experiment to be called by user
"""
    function assess_practical_identifiability(prm_keys::Vector, hprm::Hyperprm;
        t_fixed::Bool=false, t_end::Float64=50.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0, N::Int64=20)

Assess the practical identifiability of parameters in the Klausmeier model using multiple-restart MLE 
and Fisher information approximations.

# Arguments
    - `prm_keys::Vector`: A vector of the parameter names that are estimated by the MLE
    - `hprm::Hyperprm`: Sets true underlying parameters needed for model simulation
    - `t_fixed::Bool`: true if we consider a fixed observation time window
    - `t_end::Float64`: end of observation window (if `t_fixed=true`)
    - `t_step::Float64`: step size with which M observations should be picked (set if `t_fixed=false`)
    - `obs_late::Bool=false`: Set to true if we only consider observations taken in the stable state
    - `t_obs::Float64=100.0`: The time at which late observations are taken if `obs_late=true`
    - `N::Int64=20`: Number of random restarts for the MLE procedure

# Behavior
    1. Simulates the Klausmeier model with the given hyperparameters and generates noisy observations
    2. Performs multiple-restart MLE to estimate the parameters specified in `prm_keys`
    3. Computes the covariance and correlation matrices based on the estimated MLE
    4. Visualizes results:
        - Parameter estimates across MLE restarts
        - Losses across restarts
        - Gaussian approximations of parameter uncertainties (heatmap and surface plots)

# Returns
    A named tuple with the following fields:
    - `mle`: The maximum likelihood estimate (MLE) of the parameters
    - `plot_mles`: A plot showing MLEs across multiple restarts
    - `plot_losses`: A plot showing losses across multiple restarts
    - `cor`: The correlation matrix of the estimated parameters
    - `cov`: The covariance matrix of the estimated parameters
    - `gaussian_heatmap`: Heatmap of the Gaussian (Fisher) approximation of parameter uncertainties
    - `gaussian_surface`: Surface plot of the Gaussian (Fisher) approximation of parameter uncertainties
 """       
function assess_practical_identifiability(prm_keys::Vector, hprm::Hyperprm;
    t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0, N::Int64=20)
    
    # create data observations and include noise
    sol_true = sol_klausmeier(hprm; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)
    randomize_data!(sol_true, hprm.noise); # make data noisy

    # compute MLE via multiple restart optimizations
    inits, inits_loss, mles, losses, best_loss_ind, converged = mult_restart_mle(N, prm_keys, hprm, sol_true, t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)
    mle = mles[best_loss_ind,:]
    plot_mles = plot_mult_restart_mles(inits, mles, best_loss_ind, prm_keys, hprm, N=N)
    plot_losses = plot_mult_restart_losses(inits_loss, losses,best_loss_ind, N=N)

    # compute covariance/ correlation Matrix
    cor, cov = correlation_covariance_matrix(mle, prm_keys, hprm, sol_true, t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)

    # plot MvNormals (Fisher Approximation)
    gaussian_heatmap, gaussian_surface = plot_gaussian(mle,cov,prm_keys)

    return return (
        mle = mle,
        plot_mles = plot_mles,
        plot_losses = plot_losses,
        cor = cor,
        cov = cov,
        gaussian_heatmap = gaussian_heatmap,
        gaussian_surface = gaussian_surface,
    )
end

"""
    function analyze_ll(mle::Vector, prm_keys::Vector, hprm_true::Hyperprm, cutoff::Int64; 
                        t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, 
                        obs_late::Bool=false, t_obs::Float64=100.0)

Generate and visualize the log-likelihood surface of the Klausmeier model around the true parameters.
The function evaluates the log-likelihood over a parameter grid, applies a cutoff to filter low-probability regions,
and produces a corresponding heatmap/ surface plot highlighting the MLE.

# Arguments
    - `mle::Vector`: Maximum likelihood estimate (MLE) of the parameters
    - `prm_keys::Vector`: Names of the parameters for which the ll surface is evaluated
    - `hprm_true::Hyperprm`: True underlying parameters used to simulate data
    - `cutoff::Int64`: Minimum ll value retained for visualization; smaller values are masked
"""
function analyze_ll(mle::Vector, prm_keys::Vector, hprm_true::Hyperprm, cutoff::Int64; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0, levels::Int64=300)
    ll_data = gen_ll_evals(prm_keys, hprm_true, t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs)
    return plot_ll(ll_data, cutoff,mle, prm_keys,levels=levels)
end


# all tools needed in practical identifiability analysis
"""
    function correlation_covariance_matrix(eval_pt::Vector{Float64}, prm_keys::Vector, hprm::Hyperprm, true_val::DataFrame;
        t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)

Compute the covariance and correlation matrices of the estimated parameters using the Fisher Information Matrix (FIM) approximation at a given evaluation point.
We take the inverse of the Hessian of the negative log likelihood as approximation of the covariance matrix. The correlation matrix is obtained by normalization.

# Arguments
    - `eval_pt::Vector{Float64}`: Parameter values at which the FIM is evaluated (typically the MLE)
    - `prm_keys::Vector`: Names of the parameters corresponding to `eval_pt`
    - `hprm::Hyperprm`: Hyperparameters of the model (true underlying parameters)
    - `true_val::DataFrame`: Observed data used to compute the log-likelihood

# Returns
    A tuple `(cor, cov)`:
    - `cor`: Correlation matrix of the parameters at `eval_pt`
    - `cov`: Covariance matrix of the parameters at `eval_pt`
"""
function correlation_covariance_matrix(eval_pt::Vector{Float64}, prm_keys::Vector, hprm::Hyperprm, true_val::DataFrame; t_fixed::Bool=false, t_end::Float64=100.0, t_step::Float64=1.0, obs_late::Bool=false, t_obs::Float64=100.0)
    fim = - ForwardDiff.hessian(x -> compute_ll(x, prm_keys, hprm, true_val; t_fixed=t_fixed, t_end=t_end, t_step=t_step, obs_late=obs_late, t_obs=t_obs), eval_pt)
    cov = inv(fim)
    cor = [cov[i,j] / sqrt(cov[i,i]*cov[j,j]) for i in range(1, size(cov,1)), j in range(1, size(cov,2))]
    return cor, cov
end

"""
    function plot_mult_restart_mles(inits::Matrix, mles::Matrix, ind_best::Int64, prm_keys::Vector, hprm::Hyperprm; compare::Bool=true, N::Int64=20)

Plot the results of multiple-restart MLEs for given parameters, highlighting the best estimate, initial guesses, and optionally the true parameter values.

# Arguments
    - `inits::Matrix`: Initial parameter values used in the multiple restart MLEs (size N × n_prms)
    - `mles::Matrix`: Parameter estimates from each restart (size N × n_prms)
    - `ind_best::Int64`: Index of the best estimate (lowest loss)
    - `prm_keys::Vector`: Names of the parameters corresponding to the columns in `inits` and `mles`
    - `compare::Bool=true`: If true, plots initial values alongside the MLEs
    - `N::Int64=20`: Number of restarts
"""
function plot_mult_restart_mles(inits::Matrix, mles::Matrix, ind_best::Int64, prm_keys::Vector, hprm::Hyperprm; compare::Bool=true, N::Int64=20)

    n_prms = length(prm_keys)
    
    if n_prms == 1
        p = plot(mles[:,1], label="MLEs $(prm_keys[1])", title="", color="#165DB1",linewidth=2, ylabel="", xlabel="restart index")
        if hprm !== nothing && hasproperty(hprm, prm_keys[1])
            hline!([getproperty(hprm, prm_keys[1])], linestyle=:dash, linewidth=2, color=:black, label="true parameter")
        end
        scatter!(1:N, mles[:,1], markershape=:square, markersize=2, color="#165DB1", label="")
        if compare
            plot!(inits[:,1], label="inits", color="#9ABCE4",linewidth=1)
            scatter!(1:N, inits[:,1], markershape=:square, markersize=2, color="#9ABCE4", label="")
        end
        scatter!([ind_best], [mles[ind_best, 1]], markershape=:x, markerstrokewidth=5, markersize=8, color="#F7811E", label="best estimate")
        return [p]
    else
        # subplot 1
        p1 = plot(mles[:,1], label="MLEs $(prm_keys[1])", title="", color="#165DB1", linewidth=2, ylabel="", xlabel="restart index")
        if hprm !== nothing && hasproperty(hprm, prm_keys[1])
            hline!([getproperty(hprm, prm_keys[1])], linestyle=:dash, linewidth=2, color=:black, label="true parameter")
        end
        scatter!(1:N, mles[:,1], markershape=:square, markersize=2, color="#165DB1", label="")
        if compare
            plot!(inits[:,1], label="inits", color="#9ABCE4",linewidth=1)
            scatter!(1:N, inits[:,1], markershape=:square, markersize=2, color="#9ABCE4", label="")
        end
        scatter!([ind_best], [mles[ind_best, 1]], markershape=:x, markerstrokewidth=5, markersize=8, color="#F7811E", label="best estimate")
        
        # subplot 2
        p2 = plot(mles[:,2], label="MLEs $(prm_keys[2])", title="", color="#165DB1", linewidth=2, ylabel="", xlabel="restart index")
        if hprm !== nothing && hasproperty(hprm, prm_keys[2])
            hline!([getproperty(hprm, prm_keys[2])], linestyle=:dash, linewidth=2, color=:black, label="true parameter")
        end
        scatter!(1:N, mles[:,2], markershape=:square, markersize=2, color="#165DB1", label="")
        if compare
            plot!(inits[:,2], label="inits", color="#9ABCE4",linewidth=1)
            scatter!(1:N, inits[:,2], markershape=:square, markersize=2, color="#9ABCE4", label="")
        end
        scatter!([ind_best], [mles[ind_best, 2]], markershape=:x, markerstrokewidth=5, markersize=8, color="#F7811E", label="best estimate")
        #return plot(p1, p2, layout=(1,2), size=(800,400), bottom_margin=5mm)
        return [p1,p2]
    end
end

"""
    function plot_mult_restart_losses(inits_loss::Vector, losses::Vector, ind_best::Int64; compare::Bool=false, N::Int64=20)

Plot the evolution of the loss values across multiple-restart MLEs, highlighting the lowest loss and optionally the initial loss values.

# Arguments
    - `inits_loss::Vector`: Initial loss values corresponding to the starting guesses
    - `losses::Vector`: Loss values obtained from each MLE restart
    - `ind_best::Int64`: Index of the lowest loss (best estimate)
    - `compare::Bool=false`: If true, plots the initial loss values alongside the MLE losses
    - `N::Int64=20`: Number of restarts
"""
function plot_mult_restart_losses(inits_loss::Vector, losses::Vector, ind_best::Int64; compare::Bool=false, N::Int64=20)
    plot(losses, label="MLEs", color="#165DB1",linewidth=2, title="", ylabel = "loss value", xlabel="restart index")
    scatter!(1:N, losses, markershape=:square, markersize=2, color="#165DB1", label="")

    if compare
        plot!(inits_loss, label="init", color="#9ABCE4",linewidth=1)
        scatter!(1:N, inits_loss, markershape=:square, markersize=2, color="#9ABCE4", label="")
    end

    scatter!([ind_best], [losses[ind_best]], markershape=:x, markerstrokewidth=5, markersize=8, color="#F7811E", label="lowest")
end

"""
    function plot_gaussian(mle::Vector, cov::Matrix, prm_keys::Vector)

Visualize the Gaussian (Fisher) approximation of parameter uncertainties based on the MLE and covariance matrix. Supports 1D and 2D parameter cases.

# Arguments
    - `mle::Vector`: Maximum likelihood estimate of the parameters
    - `cov::Matrix`: Covariance matrix of the parameters
    - `prm_keys::Vector`: Names of the parameters

# Behavior
    - For a single parameter, plots the 1D normal distribution curve
    - For two parameters, plots a 2D heatmap and 3D surface plots of the joint distribution

# Returns
    - `heatmap_plot`: Heatmap of the Gaussian density (2D case) or `nothing` (1D case)
    - `surface_plot`: 1D line plot (1D case) or side-by-side 3D surface plots (2D case)
"""
function plot_gaussian(mle::Vector, cov::Matrix, prm_keys::Vector)

    if length(mle) == 1
        dens = Normal(mle[1], sqrt(cov[1]))
        x = range(mle[1] - 3*sqrt(cov[1,1]), mle[1] + 3*sqrt(cov[1,1]), length=1000)
        pdf_evals = pdf.(dens, x)
        surface_plot = plot(x,pdf_evals, xlabel=prm_keys[1], linewidth=2, color="#165DB1", label="")
        heatmap_plot = nothing
    else
        # set custom color gradient
        tum_blues = ["#D7E4F4", "#C2D7EF", "#9ABCE4", "#5E94D4", "#165DB1", "#14519A", "#114584", "#0E396E"]
        tum_cgrad = cgrad(tum_blues, categorical=false)

        cov = Symmetric((cov + cov') / 2) # ensure numerical stability
        dens = MvNormal(mle, cov)
        
        # plotting ranges
        d = max(3*sqrt(cov[1,1]),3*sqrt(cov[2,2])) # make axes comparable
        a = range(mle[1]-d, mle[1]+d, length=1000)
        m = range(mle[2]-d, mle[2]+d, length=1000)
        pdf_evals = [pdf(dens, [ai, mi]) for mi in m, ai in a]

        # create heatmap
        heatmap_plot = heatmap(a,m,pdf_evals, xlabel=prm_keys[1], ylabel=prm_keys[2], color=tum_cgrad, colorbar=false)

        # create 3D plots from different viewpoints
        v1 = plot(a, m, pdf_evals, st=:surface, xlabel=prm_keys[1], ylabel=prm_keys[2], color=tum_cgrad, colorbar=false)
        v2 = plot(m, a, pdf_evals', st=:surface, xlabel=prm_keys[2], ylabel=prm_keys[1], color=tum_cgrad, colorbar=false)
        surface_plot = [v1,v2]
    end

    return heatmap_plot, surface_plot
end