
using Pkg
Pkg.activate("ClusterPolicyGen")
Pkg.instantiate()

using POMDPModels
using SARSOP
using POMDPs


# hyperparameters
# mutable struct TigerPOMDP <: POMDP{Bool, Int64, Bool}
#     r_listen::Float64
#     r_findtiger::Float64
#     r_escapetiger::Float64
#     p_listen_correctly::Float64
#     discount_factor::Float64
# end
# TigerPOMDP() = TigerPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95)


# r_listen::Float64 = -1.0: 
    # This is the reward (actually a cost, since it's negative) 
    # incurred whenever the agent chooses to listen. Listening 
    # provides information at a small penalty.
# r_findtiger::Float64 = -100.0: 
    # This is the large negative reward (punishment) if the agent 
    # opens the door with the tiger behind it. Opening the wrong 
    # door is very costly.
# r_escapetiger::Float64 = 10.0: 
    # This is the positive reward for opening the door with the 
    # treasure (no tiger). Opening the correct door yields this 
    # positive reward.
# p_listen_correctly::Float64 = 0.85: 
    # This is the probability that listening correctly reveals 
    # the tiger's location. When listening, the observation is 
    # correct with this probability and incorrect with probability 
    # 1 - 0.85 = 0.15.
# discount_factor::Float64 = 0.95: 
    # The standard POMDP discount factor for future rewards, ensuring
    # the value function converges and providing a planning horizon.


pomdp = TigerPOMDP()
solver = SARSOPSolver(verbose=true, timeout=200)
policy = solve(solver,pomdp)
mv("policy.out", "policy$λ.out")


#     r_listen::Float64
#     r_findtiger::Float64
#     r_escapetiger::Float64
#     p_listen_correctly::Float64
#     discount_factor::Float64


mean(policy.alphas)
policy.alphas
# Suppose alpha_matrix is your n_alpha × 303 matrix
# Each row is an alpha vector, each column is a state

# Example: create a random matrix for demonstration
alpha_matrix = randn(50, 303)  # 50 alpha vectors, 303 states

# Compute mean alpha vector (length 303)
mean_alpha = mean(alpha_matrix, dims=1)  # returns a 1×303 array

# Compute variance per state (length 303)
var_alpha = var(alpha_matrix, dims=1)    # returns a 1×303 array

# To get as a flat vector:
mean_alpha_vec = vec(mean_alpha)
var_alpha_vec = vec(var_alpha)

println("Mean alpha vector (first 5 states): ", mean_alpha_vec[1:5])
println("Variance per state (first 5 states): ", var_alpha_vec[1:5])

# cosine similarity first
# then plot means and variances over plots
# try autoencoders?