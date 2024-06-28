include("get_data.jl")
include("GLV.jl")
using JLD2, Plots, StatsPlots, Statistics
using PyPlot, HypothesisTests

# Get mean and std of interaction matrix A and μ
path_name = "."
As = []
μs = []

# For simplicity, we will only consider experiments that has 35 passages.
passage_data_size = size(passage_data, 3)
show_idxs = passage_data_size
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
    truth_nan = .!isnan.(truth)
    truth = truth[truth_nan]
    preds_mean = passage_x[:, :, end][truth_nan]
    passage_corr = pcorr(truth, preds_mean)
    println(passage_corr)

    # We want to "passage" dynamics to filter failed cases
    passage_x2, passage_t = passage_GLV(passage_data[:, :, 1], pars, dt, passage_tf, Int(passage_iters), num_passages, dilution_factor)
    if passage_corr < 0.85 || any(passage_x .> 1e2) || any(passage_x2 .> 1e2) || isnan(passage_corr) # Filter models that succeeded
        continue
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

# Save the plot as an SVG file
plt.savefig(path_name * "//EXP0004_P_passage_constr.svg")

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
CSV.write(path_name * "//EXP0004_P_passage_constr.csv", df_pars)
