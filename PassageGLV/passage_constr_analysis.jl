include("get_data.jl")
include("GLV.jl")
using JLD2, DataFrames, Statistics, PyPlot, ColorSchemes, Colors, Plots

# Check original passage data performance
df = DataFrame(
    model_num = Int[],
    mse = Float64[],
    mses = Vector{Float64}[],
    corr = Float64[],
    corrs = Vector{Float64}[],
)
preds = []
passage_data_size = size(passage_data, 3)
show_idxs = passage_data_size

# For simplicity, we will only consider experiments that has 35 passages.
path_name = "."
for (idx, names) in enumerate(readdir(path_name * "//GLV_pars_passage_constr"))
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

    # Compute feature-wise correlation
    passage_corrs = zeros(size(passage_data, 1))
    passage_mses = zeros(size(passage_data, 1))
    for i in axes(passage_data, 1)
        truth = passage_data[i, :, end]
        truth_idx = .!isnan.(truth) .&& .!iszero.(truth)
        truth = truth[truth_idx]
        preds_mean = passage_x[i, :, end][truth_idx]
        try
            passage_corrs[i] = pcorr(truth, preds_mean)
            passage_mses[i] = mse_loss(truth, preds_mean)
        catch e
            println(e)
        end
    end

    # We want to "passage" dynamics to filter failed cases
    passage_x2, passage_t = passage_GLV(passage_data[:, :, 1], pars, dt, passage_tf, Int(passage_iters), num_passages, dilution_factor)
    if passage_corr < 0.85 || any(passage_x .> 1e2) || any(passage_x2 .> 1e2) || isnan(passage_corr) # Filter models that succeeded
        continue
    end
    push!(preds, passage_x)
    push!(df, (idx, passage_mse, passage_mses, passage_corr, passage_corrs))
end
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

# Save dataframe
CSV.write(path_name * "//EXP0004_passage_constr.csv", pred_df)
