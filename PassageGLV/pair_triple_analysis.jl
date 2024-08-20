include("get_data.jl")
include("GLV.jl")
using JLD2, Plots, StatsPlots, Statistics
using PyPlot, HypothesisTests, ProgressMeter

# Check original passage data performance
passage_data_size = size(passage_data, 3)

# Record predictions
pred_data = Dict()
num_models = zeros(size(passage_data, 2))
failed_models = 0
total_models = 0
path_name = "."
filename = "GLV_pars_pair"
savefile = "EXP0004_pair_analysis.csv"
savefile2 = "EXP0004_pair_analysis_params.csv"
# filename = "GLV_pars_triple"
# savefile = "EXP0004_triple_analysis.csv"
# savefile2 = "EXP0004_triple_analysis_params.csv"
As = []
μs = []
ProgressMeter.@showprogress for (idx, names) in enumerate(readdir(path_name * "//" * filename))
    global failed_models, total_models, pred_data, num_models
    # Model fit settings
    dt = 3/12 # time step is 3/12 or 0.25
    passage_tf = 24.0
    passage_iters = passage_tf/dt
    cur_data = JLD2.load(path_name * "//" * filename * "//" * names)
    pars = cur_data["GLV_pars"]
    train_idxs = cur_data["train_idxs"]
    valid_idxs = cur_data["valid_idxs"]

    # Assess training performance
    passage_x, passage_t = passage_GLV(passage_data[:, train_idxs, 1], pars, dt, Int(passage_iters), num_passages, dilution_factor)
    
    # Get truths and predictions. No nans, exlucde initial point
    truth = passage_data[:, train_idxs, end]
    truth_idx = .!isnan.(truth) .&& .!iszero.(truth)
    truth = truth[truth_idx]
    preds_mean = passage_x[:, :, end][truth_idx]
    passage_corr = pcorr(truth, preds_mean)
    passage_mse = mse_loss(truth, preds_mean)

    # We want to "passage" dynamics to filter failed cases. Unlike the full case, we need a less stringent filter (Too many failed cases)
    passage_x2, passage_t = passage_GLV(passage_data[:, train_idxs, 1], pars, dt, passage_tf, Int(passage_iters), num_passages, dilution_factor)
    if passage_corr < 0.8 || any(passage_x .> 1e2) || any(passage_x2 .> 1e2) || isnan(passage_corr) # Filter models that succeeded
        failed_models += 1
        total_models += 1
        continue
    end
    total_models += 1

    # Attempt to predict the held out data
    passage_x, passage_t = passage_GLV(passage_data[:, valid_idxs, 1], pars, dt, Int(passage_iters), num_passages, dilution_factor)
    
    # Skip if the model failed
    if any(passage_x .> 1e2)
        failed_models += 1
        continue
    end
    
    # Store the predictions
    for (idx, idx_pred) in enumerate(valid_idxs)
        if idx_pred ∈ keys(pred_data)
            pred_data[idx_pred] = cat(pred_data[idx_pred], passage_x[:, idx:idx, :], dims=2)
        else
            pred_data[idx_pred] = passage_x[:, idx:idx, :]
        end
    end

    # Get A and μ
    num_sp = size(passage_data, 1)
    A = reshape(pars[1:num_sp*num_sp], num_sp, num_sp)
    μ = reshape(pars[num_sp*num_sp+1:end], num_sp)
    push!(As, A)
    push!(μs, μ)
end
A_mean = mean(As)
A_std = std(As)
μ_mean = mean(μs)
μ_std = std(μs)
println("Failed models: ", failed_models)
println("Successful models: ", total_models - failed_models)
println("Failed models Percentage: ", failed_models/total_models*100.0)
display(length(keys(pred_data)))

# Take the mean and std of the predictions
all_preds_mean = zeros(size(passage_data, 1), size(passage_data, 2), size(passage_data, 3))
all_preds_std = zeros(size(passage_data, 1), size(passage_data, 2), size(passage_data, 3))
for keys in keys(pred_data)
    all_preds_mean[:, keys, :] = mean(pred_data[keys], dims=2)
    all_preds_std[:, keys, :] = std(pred_data[keys], dims=2)
end

# Go through every sample
pred_df = DataFrame()
for i in axes(passage_data, 2)
    global pred_df, pred_data
    # Skip mono cultures
    if i ∉ keys(pred_data)
        continue
    end

    # Get treatment name
    cur_species = findall((x) -> x, .!passage_u1[:, i, 1])
    cur_species = [species_names[cur_sp] for cur_sp in cur_species]
    treatment = ""
    for (idx2, sp_str) in enumerate(cur_species)
        if idx2 < length(cur_species)
            treatment = treatment * sp_str * "_"
        else
            treatment = treatment * sp_str
        end
    end

    for j in axes(passage_data, 3)
        if any(isnan.(passage_data[:, i, j]))
            continue
        end

        _df = DataFrame(
            "Treatments" => String[],
            "species" => String[],
            "true" => Float64[],
            "pred" => Float64[],
            "stdv" => Float64[],
            "n" => Int[],
            "passage" => Int[],
        )
        for (k, sp_str) in enumerate(species_names)
            sp_str = species_names[k]
            push!(_df, (treatment, sp_str, passage_data[k, i, j], all_preds_mean[k, i, j], all_preds_std[k, i, j], num_models[i], j))
        end

        # Push the dataframe row to the main dataframe
    pred_df = vcat(pred_df, _df)
    end
end

# Get only passage 7
unique(pred_df.passage)
passage_7_df = pred_df[pred_df.passage .== unique(pred_df.passage)[4], :]

# Compute the correlation
true_y = passage_7_df[:, "true"]
pred_y = passage_7_df[:, "pred"]

# Remove zeros
not_zeros = .!iszero.(true_y) .& .!iszero.(pred_y)
true_y_rz = true_y[not_zeros]
pred_y_rz = pred_y[not_zeros]

# Compute the correlation
correlation = cor(true_y_rz, pred_y_rz)
display(correlation)
p = Plots.scatter(true_y_rz, pred_y_rz, seriestype = :scatter, legend=false, xlabel="True", ylabel="Predicted", title="Correlation: $correlation")
display(p)

# Save dataframe
CSV.write(path_name * "//" * savefile, pred_df)

# Create a figure and subplots with a larger size
fig, axs = subplots(size(A_mean)..., figsize=(14, 14))  # Adjust the figsize as needed

# Plot Gaussian fit for each entry in A
x = range(-1.0, 1.0, length=100)
for i in axes(A_mean, 1)
    for j in axes(A_mean, 2)
        # Generate x values for the Gaussian curve
        left_val = -5.0
        if i == j
            right_val = 0.0
        else
            right_val = 5.0
        end
        
        # Compute the Gaussian curve
        x = range(left_val, right_val, length=100)
        y = 1.0 / (A_std[i, j] * sqrt(2π)) * exp.(-((x .- A_mean[i, j]) .^ 2) ./ (2.0 * A_std[i, j]^2))

        # Get appropriate bounds
        while max(y[1], y[end]) > 0.01
            left_val -= 1.0
            right_val += 1.0

            # Compute the Gaussian curve
            x = range(left_val, right_val, length=100)
            y = 1.0 / (A_std[i, j] * sqrt(2π)) * exp.(-((x .- A_mean[i, j]) .^ 2) ./ (2.0 * A_std[i, j]^2))
        end
        
        # Plot the Gaussian curve
        axs[i, j].plot(x, y, color="blue")

        # Conduct t-test
        t_test_result = OneSampleTTest(A_mean[i, j], A_std[i, j], length(As), 0.0)

        # Add star to the plot if the null hypothesis is rejected
        if pvalue(t_test_result) < 0.05
            axs[i, j].text(0.05, 0.95, "*", transform=axs[i, j].transAxes, fontsize=20, color="red")
        end
        
        # Set x-axis label only for the bottom row
        if i == size(A_mean, 1)
            axs[i, j].set_xlabel(species_names[j])
        end
        
        # Set y-axis label only for the leftmost column
        if j == 1
            axs[i, j].set_ylabel(species_names[i])
        end
    end
end

# Adjust the spacing between subplots
fig.tight_layout()

# Display the plot
display(fig)
plt.close()

# Save parameters. Note that it is from j to i or j -> i!
df_pars = DataFrame()
for i in axes(μ_mean, 1)
    interaction_name = species_names[i]
    t_test_result = OneSampleTTest(μ_mean[i], μ_std[i], length(As), 0.0)
    push!(df_pars, Dict(
        "Param name" => interaction_name, 
        "Param value" => μ_mean[i], 
        "Param stdv" => μ_std[i],
        "Param p-value" => pvalue(t_test_result)
    ), cols=:union)
end

A_pvalues = zero.(A_mean)
for i in axes(A_mean, 1), j in axes(A_mean, 2)
    interaction_name = species_names[i]*"*"*species_names[j]
    t_test_result = OneSampleTTest(A_mean[i, j], A_std[i, j], length(As), 0.0)
    push!(df_pars, Dict(
        "Param name" => interaction_name, 
        "Param value" => A_mean[i, j], 
        "Param stdv" => A_std[i, j],
        "Param p-value" => pvalue(t_test_result)
    ), cols=:union)
    A_pvalues[i, j] = pvalue(t_test_result)
end

# Save dataframe
CSV.write(path_name * "//" * savefile2, df_pars)
