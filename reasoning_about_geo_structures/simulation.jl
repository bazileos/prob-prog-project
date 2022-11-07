using PyCall
using Distributions
using Gen
include("./model.jl")

# print(np.linspace(2000,2020,1000))
nr_of_samples = 100
t = collect(range(2000, 2020, length=nr_of_samples))
head = map((x) -> (x-t[1])*-1+5*cos(pi*2*x), t)
println(t)
println(head)

trace = Gen.simulate(geo_model, (head,t))
choices = Gen.get_choices(trace)

o = Vector{Float32}(undef, nr_of_samples)
for s=1:nr_of_samples
    o[s] =  choices[(:obs, s)]
end


touch("./reasoning_about_geo_structures/data.txt")
file = open("./reasoning_about_geo_structures/data.txt", "w")
for s=1:nr_of_samples
    time_stamp = t[s]
    if s == nr_of_samples
        write(file, "$time_stamp")
    else
        write(file, "$time_stamp\t")
    end
end
write(file, "\n")
for s=1:nr_of_samples
    head_val = head[s]
    if s == nr_of_samples
        write(file, "$head_val")
    else
        write(file, "$head_val\t")
    end
end
write(file, "\n")
for s=1:nr_of_samples
    if s == nr_of_samples
        obs = choices[(:obs, s)]
        write(file, "$obs")
    else 
        obs = choices[(:obs, s)]
        write(file, "$obs\t")
    end
end
close(file)
Kv = choices[:Kv]
Sskv = choices[:Sskv]
Sske = choices[:Sske]
nclay = choices[:nclay]

println("SIMULATION COMPLETED")
println("Kv: ", Kv)
println("Sskv: ", Sskv)
println("Sske: ", Sske)
println("nclay: ", nclay)

touch("./reasoning_about_geo_structures/parameters.txt")
file = open("./reasoning_about_geo_structures/parameters.txt", "w")
write(file, "Kv: $Kv\n")
write(file, "Sskv: $Sskv\n")
write(file, "Sske: $Sske\n")
write(file, "nclay: $nclay\n")
close(file)

# Sskv ~ cauchy(-3.5, 3) #m-1
# Sske ~ cauchy(-5, 3) #m-1
# nclay ~ uniform(5, .., 10)
# claythick=5 # m
# head = [provided as input]
# reference_time = [provided as input]
# # run simulation
# t,defm,head,defm_v=calc_deformation(t,head,10**Kv,10**Sskv,10**Sske,claythick,nclay)
# aligned_deformation=numpy.interp(reference_time,time,defm)
# observe(aligned_deformation, Normal(observed_deformation, 2))