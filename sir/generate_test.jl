include("./model.jl")
using Gen

trace = simulate(sir_model, ())
choices = get_choices(trace)

println("tau: $(choices[:tau]), R0: $(choices[:R0]), rho0: $(choices[:rho0]), rho1: $(choices[:rho1]), rho2: $(choices[:rho2]), switch_to_rho1: $(choices[:switch_to_rho1]), switch_to_rho2: $(choices[:switch_to_rho2])")

touch("parameters.txt")
file = open("parameters.txt", "w")
write(file, "tau: $(choices[:tau])\n")
write(file, "R0: $(choices[:R0])\n")
write(file, "rho0: $(choices[:rho0])\n")
write(file, "rho1: $(choices[:rho1])\n")
write(file, "rho2: $(choices[:rho2])\n")
write(file, "switch_to_rho1: $(choices[:switch_to_rho1])\n")
write(file, "switch_to_rho2: $(choices[:switch_to_rho2])\n")
close(file)

touch("data.txt")
file = open("data.txt", "w")
for t = 2:60
    o_t = choices["o_$t"]
    write(file, "$t\t$o_t\n")
end
close(file)