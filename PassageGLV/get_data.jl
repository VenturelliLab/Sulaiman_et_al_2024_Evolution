using CSV, DataFrames, Statistics

# Passage
df = DataFrame(CSV.File("EXP0004_35passages_formodelfitting.csv"))
species_names = df[:, Not([:Time, :Sample, :Replicate])] |> names
species_names = Dict(zip(1:length(species_names), species_names))

# Get sample column unique values
sample_keys = unique(df[:, "Sample"])

# Construct a matrix for data for every sample and replicate
expected_time_arr = range(0.0, 35.0, length=36) |> collect 
num_sp = 10
data = []
for sample in sample_keys
    # Get data for current sample
    sample_data = df[df[:, "Sample"] .== sample, :]

    # Combine replicates by mean
    sample_data = combine(groupby(sample_data, :Time), Not([:Time, :Sample, :Replicate]) .=> mean)

    # Gather data
    tmp_data_mat = []
    for t in expected_time_arr
        # Get data for current time
        time_data = sample_data[sample_data[:, "Time"] .== t, :]
        cur_data = time_data[:, Not([:Time])]
        cur_data = Array(cur_data)
        if size(time_data, 1) == 0 # If no data, fill with NaN
            cur_data = NaN .* ones(1, num_sp)
        end
        # Push to array
        push!(tmp_data_mat, cur_data)
    end
    tmp_data_mat = Matrix(reduce(vcat, tmp_data_mat)') # Features first
    push!(data, tmp_data_mat)
end

# Complete data
passage_data = reduce((x, y) -> cat(x, y; dims=3), data)
passage_data = permutedims(passage_data, [1, 3, 2]) # Samples first
passage_data = ifelse.(passage_data .< 0.0, 0.0, passage_data)
num_passages = size(passage_data, 3) - 1
dilution_factor = 40.0
passage_u1 = cat([iszero.(passage_data[:, :, 1]) for idx in axes(passage_data, 3)]...; dims=3)
