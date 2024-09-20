using Distributed
if nprocs() <= 9 # 1 main process and 8 other processes
    addprocs(9 - nprocs())
end

@everywhere begin
    include("get_data.jl")
    include("GLV.jl")
    using Random, Optimization, OptimizationNLopt, JLD2
end

# Easiest way to train multiple models, may not be the most efficient though.
path_name = "GLV_pars_passage_constr"
@distributed for i = 1:64 # Note that i is not used to save anything
    # Model fit settings
    Random.seed!(i) # Try fixing the seed, but it may be ignored
    dt = 3/12 # time step is 3/12 or 0.25
    passage_tf = 24.0
    passage_iters = passage_tf/dt

    # Nonlinear optimization objective
    f = p -> evaluate_GLV(
        p, 
        Int(passage_iters), num_passages, passage_data, dilution_factor,
        dt,
        1e-6, 0.5 # Elastic net regularization
    )[1]

    # Constraints and initial guess
    num_sp = size(passage_data, 1)
    lower_finite = []
    upper_finite = []

    # Interactions
    for i in 1:num_sp, j in 1:num_sp
        if i == j
            push!(lower_finite, -20.0)
            push!(upper_finite, 0.0)
        else
            push!(lower_finite, -20.0)
            push!(upper_finite, 20.0)
        end
    end

    # Set growth rates: CD, DP, CA, PC, CS, CH, BU, BT, PV, EL
    μ_vector = [0.64, 0.52, 0.63, 0.23, 0.31, 0.45, 0.92, 0.77, 0.84, 0.43]

    # Growth rates
    for i in 1:num_sp
        push!(lower_finite, μ_vector[i])
        push!(upper_finite, μ_vector[i])
    end

    # Initial guess
    A = randn(num_sp, num_sp)
    A = A - Diagonal(A) - 5.0*Diagonal(abs.(A)) .- 1.0 # Negative diagonal
    μ = μ_vector
    initial_x = vcat(A[:], μ)

    # Optimize
    callback = (opt_state, loss_val) -> begin
        println("Process: $i Current loss: $loss_val")
        return false
    end
    f2 = OptimizationFunction((x, p) -> f(x))
    prob = Optimization.OptimizationProblem(f2, initial_x, Float64[], lb = Vector{Float64}(lower_finite), ub = Vector{Float64}(upper_finite))
    sol = solve(
        prob, NLopt.G_MLSL_LDS();
        population = 32,
        callback = callback, 
        maxtime = 1800, 
        local_method = NLopt.LN_SBPLX() # More robust nelder-mead
    ) 

    # Save solutions. If file exists, try again
    filename = path_name * "//GLV_pars_" * randstring(5) * ".jld2"
    tries = 10 # Try 10 times
    cur_try = 0
    while isfile(filename) && cur_try < tries
        filename = path_name * "//GLV_pars_" * randstring(5) * ".jld2"
        cur_try += 1
    end
    JLD2.save(
        filename, 
        "GLV_pars", sol.u, 
        "best_loss", f(sol.u), 
    )
end
