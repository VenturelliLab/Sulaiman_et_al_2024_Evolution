using Distributed
# Change this as necessary
if nprocs() <= 11 # 1 main process and 10 other processes
    addprocs(11 - nprocs())
end

@everywhere begin
    include("get_data.jl")
    include("GLV.jl")
    using Random, Optimization, OptimizationNLopt, JLD2
end

path_name = "GLV_pars_passage_constr_LOOCV"
repetitions = 10 # Do 25 repetitions everytime (original)
comm_sizes = sum(passage_data[:, :, 1] .> 0.0; dims=1)
comm_idxs = 1:size(passage_data, 2) |> collect
is_not_mono = comm_idxs[comm_sizes[:] .> 1]

# Remaining ones specified (as needed). May have to use different seeds until it works
remaining_idxs = [30 34 38 42 46] # [30 34 38 42 46]
remain_base_seed = 4500 # base seed int for the remaining ones, usually is 0, 2000, 4000, 4500.

# Ideally, run until we have length(keys(pred_data)) == length(is_not_mono)
# We need to run this until we have all the samples and each has at least 3 repetitions
for idx in remaining_idxs # Only validate non-monoculture samples (full run) OR use remaining_idxs
    @sync @distributed for rep = 1:repetitions # Tweak the starting index if already done. Sync is important to not start all at once
        # Model fit settings
        Random.seed!(idx + (rep - 1)*size(passage_data, 2) + remain_base_seed) # Try fixing the seed, but it may be ignored
        dt = 3/12 # time step is 3/12 or 0.25
        passage_tf = 24.0
        passage_iters = passage_tf/dt

        # Leave-one-out CV
        sample_idxs = 1:size(passage_data, 2)
        sample_idxs = sample_idxs[sample_idxs .!= idx]

        # Nonlinear optimization objective
        f = p -> evaluate_GLV(
            p, 
            Int(passage_iters), num_passages, passage_data[:, sample_idxs, :],
            dilution_factor,
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

        # Initial guess. Bound stuff since number of iterations is high and 
        # there is a good chance where A has a bad initialization
        A = max.(min.(randn(num_sp, num_sp), 19.0), -19.0)
        A = A - Diagonal(A) - 5.0*Diagonal(abs.(A)) .- 1.0 # Negative diagonal
        μ = μ_vector
        initial_x = vcat(A[:], μ)

        # Optimize
        callback = (opt_state, loss_val) -> begin
            println("Rep: $rep Set: $idx Current loss: $loss_val")
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

        # Save solutions. If file exists, try again. If we are unlucky, we will overwrite
        filename = path_name * "//GLV_pars_" * randstring(8) * ".jld2"
        tries = 10 # Try 10 times
        cur_try = 0
        while isfile(filename) && cur_try < tries
            filename = path_name * "//GLV_pars_" * randstring(8) * ".jld2"
            cur_try += 1
        end
        JLD2.save(
            filename,
            "GLV_pars", sol.u, 
            "best_loss", f(sol.u), 
            "idx", idx
        )
    end
end
