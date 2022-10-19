include("./inference.jl")
file_name = ARGS[1]
f = open(file_name, "r")
o ::Vector{Int} = zeros(0)
for line in readlines(f)
    append!(o, parse(Int, split(line, "\t")[2]))
end
close(f)
@time unfold_pf_traces = unfold_particle_filter(5000, o, 100);
cho_res = Gen.get_choices(unfold_pf_traces)
println("tau: $(cho_res[:tau]), R0: $(cho_res[:R0]), rho0: $(cho_res[:rho0]), rho1: $(cho_res[:rho1]), rho2: $(cho_res[:rho2]), switch_to_rho1: $(cho_res[:switch_to_rho1]), switch_to_rho2: $(cho_res[:switch_to_rho2])")
