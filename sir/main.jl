include("./inference.jl")

f = open("./sir/data.txt", "r")
o ::Vector{Int} = zeros(0)
for line in readlines(f)
    append!(o, parse(Int, split(line, "\t")[2]))
end
close(f)

(tau, R0, rho0, rho1, rho2, switch_to_rho1, switch_to_rho2) = sir_inference(o, 1000)
println("tau: $tau, R0: $R0, rho0: $rho0, rho1: $rho1, rho2: $rho2, switch_to_rho1: $switch_to_rho1, switch_to_rho2: $switch_to_rho2")