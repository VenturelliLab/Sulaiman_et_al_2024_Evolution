include("get_data.jl")
include("GLV.jl")
using JLD2, DataFrames, Statistics, PyPlot, ColorSchemes, Colors, Plots

# Check original passage data performance
preds = []
passage_data_size = size(passage_data, 3)
show_idxs = passage_data_size

# For simplicity, we will only consider experiments that has 35 passages.
path_name = "."
passage_t_use = []
passage_t_index = []
for (idx, names) in enumerate(readdir(path_name * "//GLV_pars_passage_constr"))
    global passage_t_use, passage_t_index
    local passage_t, truth
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

    # We want to "passage" dynamics
    passage_x2, passage_t = passage_GLV(passage_data[:, :, 1], pars, dt, passage_tf, Int(passage_iters), num_passages, dilution_factor)
    if passage_corr < 0.85 || any(passage_x .> 1e2) || any(passage_x2 .> 1e2) || isnan(passage_corr) # Filter models that succeeded
        continue
    end
    
    passage_t_use = deepcopy(passage_t)
    push!(preds, passage_x2)
end
all_preds_mean = mean(preds)
all_preds_std = std(preds)

# Plot example passage experiment
rand_idx = rand(1:size(passage_data, 2))
truth = passage_data[:, rand_idx, :]
truth_t = 0:24:24*num_passages
fig, ax = subplots(1, 1, figsize=(6, 4))

# Plot truth
for i in axes(truth, 1)
    ax.scatter(truth_t, truth[i, :])
end

# Plot predicted
for i in axes(all_preds_mean, 1)
    bottom = all_preds_mean[i, rand_idx, :] - 1.96*all_preds_std[i, rand_idx, :] ./ sqrt.(length(preds))
    top = all_preds_mean[i, rand_idx, :] + 1.96*all_preds_std[i, rand_idx, :] ./ sqrt.(length(preds))
    ax.plot(passage_t_use, all_preds_mean[i, rand_idx, :])
    ax.fill_between(passage_t_use, bottom, top, alpha=0.5)
end

# Show plot
ax.set_xlabel("Time (Minutes)")
ax.set_ylabel("OD")
ax.set_title("EXP0004 Experiment "*string(rand_idx))

fig.tight_layout()
display(fig)
plt.close()
