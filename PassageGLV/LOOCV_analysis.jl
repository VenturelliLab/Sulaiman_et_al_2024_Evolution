include("get_data.jl")
include("GLV.jl")
using JLD2, Plots, StatsPlots, Statistics
using PyPlot, HypothesisTests, ProgressMeter

# Check original passage data performance
passage_data_size = size(passage_data, 3)

# Record predictions
pred_data = Dict()
idx_models = Dict()
num_models = zeros(size(passage_data, 2))
failed_models = 0
total_models = 0
path_name = "."
ProgressMeter.@showprogress for (idx, names) in enumerate(readdir(path_name * "//GLV_pars_passage_constr_LOOCV"))
    global failed_models, total_models, pred_data, idx_models, num_models
    # Model fit settings
    dt = 3/12 # time step is 3/12 or 0.25
    passage_tf = 24.0
    passage_iters = passage_tf/dt
    cur_data = JLD2.load(path_name * "//GLV_pars_passage_constr_LOOCV//"*names)
    pars = cur_data["GLV_pars"]
    idx_pred = cur_data["idx"]
    train_idxs = 1:size(passage_data, 2)
    train_idxs = train_idxs[train_idxs .!= idx_pred]

    # Get total models per idx
    if idx_pred ∈ keys(idx_models)
        idx_models[idx_pred] += 1
    else
        idx_models[idx_pred] = 1
    end

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
    passage_x, passage_t = passage_GLV(passage_data[:, idx_pred:idx_pred, 1], pars, dt, Int(passage_iters), num_passages, dilution_factor)
    
    # Skip if the model failed
    if any(passage_x .> 1e2)
        failed_models += 1
        continue
    end
    
    # Store the predictions
    if idx_pred ∈ keys(pred_data)
        pred_data[idx_pred] = cat(pred_data[idx_pred], passage_x, dims=2)
    else
        pred_data[idx_pred] = passage_x
    end
    num_models[idx_pred] += 1
end
println("Failed models: ", failed_models)
println("Successful models: ", total_models - failed_models)
println("Failed models Percentage: ", failed_models/total_models*100.0)
display(length(keys(pred_data)))

# Show how many samples we have for each key. If too few, go back to fit_LOOCV.jl and run on those
display([size(pred_data[keys], 2) for keys in keys(pred_data)]')

# Take the mean and std of the predictions
all_preds_mean = zeros(size(passage_data, 1), size(passage_data, 2), size(passage_data, 3))
all_preds_std = zeros(size(passage_data, 1), size(passage_data, 2), size(passage_data, 3))
for keys in keys(pred_data)
    all_preds_mean[:, keys, :] = mean(pred_data[keys], dims=2)
    all_preds_std[:, keys, :] = std(pred_data[keys], dims=2)
end
comm_sizes = sum(passage_data[:, :, 1] .> 0.0; dims=1)
comm_idxs = 1:size(passage_data, 2) |> collect
is_not_mono = comm_idxs[comm_sizes[:] .> 1]
display(is_not_mono')
display(comm_sizes[comm_sizes[:] .> 1]')

# Show which ones we are still missing. If missing, go back to fit_LOOCV.jl
missing_data_idxs = is_not_mono[is_not_mono .∉ Ref(keys(pred_data))]'
display(missing_data_idxs)
# keys(pred_data) |> collect |> transpose

# Go through every sample
pred_df = DataFrame()
for i in axes(passage_data, 2)
    global pred_df, is_not_mono
    # Skip mono cultures
    if i ∉ is_not_mono
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
CSV.write(path_name * "//EXP0004_CV_analysis.csv", pred_df)
