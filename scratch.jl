# import Pkg; Pkg.add("Random")
using BenchmarkTools
using Plots
using Distributions
using Random

# @btime
rng = MersenneTwister(1234)
n = 10000000
x = randn!(zeros(n))
u = zeros(n)
global j = 0
for val in x
    u[j] = square(sqrt(x[j] * 4 / 3 + 23 - 10))
    println(u[j])

end


my_hist = histogram(u)
plot(my_hist)
