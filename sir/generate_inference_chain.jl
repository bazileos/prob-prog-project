include("./model.jl")
include("./inference.jl")
include("./write_to_file.jl")

using Gen
using Random
Random.seed!(3)

T = 59
(trace, _) = Gen.generate(unfold_model, (T,))

choices = Gen.get_choices(trace)
println("tau: $(choices[:tau]), R0: $(choices[:R0]), rho0: $(choices[:rho0]), rho1: $(choices[:rho1]), rho2: $(choices[:rho2]), switch_to_rho1: $(choices[:switch_to_rho1]), switch_to_rho2: $(choices[:switch_to_rho2])")

write_to_file_chain(choices, T)
o = Vector{Int}(undef, T)

for t=1:T
    o[t] = choices[:chain => t => :obs]
end
println(o)


@time unfold_pf_traces = unfold_particle_filter(50000, o, 100);
cho_res = Gen.get_choices(unfold_pf_traces)
println("tau: $(cho_res[:tau]), R0: $(cho_res[:R0]), rho0: $(cho_res[:rho0]), rho1: $(cho_res[:rho1]), rho2: $(cho_res[:rho2]), switch_to_rho1: $(cho_res[:switch_to_rho1]), switch_to_rho2: $(cho_res[:switch_to_rho2])")



