using LinearAlgebra, Statistics

# This function simulates the GLV model with euler steps
function GLV(x0::AbstractArray, p::AbstractArray, dt, iters::Int)
    # x0: initial condition
    # p: parameters
    # dt: time step
    # return: x: time series, t: time points

    # Obtain parameters
    num_sp = size(x0, 1)
    A = reshape(p[1:num_sp*num_sp], num_sp, num_sp)
    μ = reshape(p[num_sp*num_sp+1:end], num_sp)

    # Simulate
    x = deepcopy(x0)
    t = [0.0]
    for i in 1:iters
        x_next = x[:, :, end] + dt .* (x[:, :, end] .* (A * x[:, :, end] .+ μ))
        x_next = ifelse.((x_next .< 0.0), 0.0, x_next)
        x = cat(x, x_next; dims=3)
        t = vcat(t, t[end] + dt)
    end
    return x, t
end

# This function gives the entire dynamics for all passages
function passage_GLV(x0::AbstractArray, p::AbstractArray, dt, iters::Int, passages::Int, dilution_factor)
    # x0: initial condition
    # p: parameters
    # dt: time step
    # return: x: time series, t: time points

    # Simulate
    x = deepcopy(x0)
    t = [0.0]
    for i in 1:passages
        if i > 1
            x_next = GLV(x[:, :, end] ./ dilution_factor, p, dt, iters)[1][:, :, end]
        else
            x_next = GLV(x[:, :, end], p, dt, iters)[1][:, :, end]
        end
        x = cat(x, x_next; dims=3)
        t = vcat(t, t[end] + 1.0)
    end
    return x, t
end

# This function gives the end points at each passage
function passage_GLV(x0::AbstractArray, p::AbstractArray, dt, tf, iters::Int, passages::Int, dilution_factor)
    # x0: initial condition
    # p: parameters
    # dt: time step
    # return: x: time series, t: time points
    
    # Simulate
    x = deepcopy(x0)
    t = [0.0]
    for i in 1:passages
        if i > 1
            x_next, t_next = GLV(x[:, :, end] ./ dilution_factor, p, dt, iters)
        else
            x_next, t_next = GLV(x[:, :, end], p, dt, iters)
        end
        x = cat(x, x_next; dims=3)
        t_next = t_next .+ tf * (i - 1.0)
        t = vcat(t, t_next)
    end
    return x, t
end

# Model fit objective
function evaluate_GLV(
    p, 
    passage_iters, num_passages, passage_data, dilution_factor,
    dt,
    λ₁, λ₂
)
    # Evaluate GLV for passage
    passage_x0 = passage_data[:, :, 1]
    passage_x, _ = passage_GLV(passage_x0, p, dt, passage_iters, num_passages, dilution_factor)

    # Compute error for passage
    passage_error = mean(abs2, (passage_x[.!isnan.(passage_data)] - passage_data[.!isnan.(passage_data)]))

    # Compute total error
    total_error = passage_error
    total_error = ifelse(isnan(total_error), Inf, total_error)

    # Return the mean of the squared errors, L1, L2 regularization, and predicted values
    return total_error + λ₁*(λ₂*sum(abs2, p) + (1.0 - λ₂)*sum(abs, p)), passage_x
end

# Correlation function
function pcorr(y, y_pred)
    if any(isnan.(y_pred))
        return NaN
    end
    return cor(y[.!isnan.(y)][:], y_pred[.!isnan.(y)][:])
end

# Mean Squared Error
function mse_loss(y, y_pred)
    return mean(abs2, y[.!isnan.(y)] - y_pred[.!isnan.(y)])
end
