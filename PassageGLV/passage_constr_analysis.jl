include("get_data.jl")
include("GLV.jl")
using JLD2, Plots, StatsPlots, Statistics
using PyPlot, HypothesisTests, ProgressMeter

# Check original passage data performance
preds = []
passage_data_size = size(passage_data, 3)

# Record predictions
path_name = "."
failed_models = 0
total_models = 0
ProgressMeter.@showprogress for (idx, names) in enumerate(readdir(path_name * "//GLV_pars_passage_constr"))
    # Model fit settings
    dt = 3/12 # time step is 3/12 or 0.25
    passage_tf = 24.0
    passage_iters = passage_tf/dt
    pars = JLD2.load(path_name * "//GLV_pars_passage_constr//"*names)["GLV_pars"]

    # Validate solutions
    passage_x, passage_t = passage_GLV(passage_data[:, :, 1], pars, dt, Int(passage_iters), num_passages, dilution_factor)
    
    # Get truths and predictions. No nans, exlucde initial point
    truth = passage_data[:, :, end]
    truth_idx = .!isnan.(truth) .&& .!iszero.(truth)
    truth = truth[truth_idx]
    preds_mean = passage_x[:, :, end][truth_idx]
    passage_corr = pcorr(truth, preds_mean)
    passage_mse = mse_loss(truth, preds_mean)

    # We want to "passage" dynamics to filter failed cases. Here, we are looking at the internal dynamics within passages.
    passage_x2, passage_t = passage_GLV(passage_data[:, :, 1], pars, dt, passage_tf, Int(passage_iters), num_passages, dilution_factor)
    if passage_corr < 0.85 || any(passage_x .> 1e2) || any(passage_x2 .> 1e2) || isnan(passage_corr) # Filter models that succeeded
        failed_models += 1
        total_models += 1
        continue
    end
    total_models += 1
    push!(preds, passage_x)
end
println("Failed models: ", failed_models)
println("Successful models: ", total_models - failed_models)
println("Failed models Percentage: ", failed_models/total_models*100.0)
all_preds_mean = mean(preds)
all_preds_std = std(preds)

# Go through every sample
pred_df = DataFrame()
for i in axes(passage_data, 2)
    global pred_df
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
            "passage" => Int[],
        )
        for (k, sp_str) in enumerate(species_names)
            sp_str = species_names[k]
            push!(_df, (treatment, sp_str, passage_data[k, i, j], all_preds_mean[k, i, j], all_preds_std[k, i, j], j))
        end

        # Push the dataframe row to the main dataframe
    pred_df = vcat(pred_df, _df)
    end
end

# Example for computing the correlation (sanity check)
passage_7_df = pred_df[pred_df.passage .== 8, :]

# Compute the correlation
true_y = passage_7_df[:, "true"]
pred_y = passage_7_df[:, "pred"]

# Remove zeros
not_zeros = .!iszero.(pred_y)
true_y_rz = true_y[not_zeros]
pred_y_rz = pred_y[not_zeros]

# Compute the correlation
correlation = cor(true_y_rz, pred_y_rz)

# Save dataframe
CSV.write(path_name * "//EXP0004_passage_constr.csv", pred_df)
