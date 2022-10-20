using PyCall
using Distributions
using Gen
include("./model.jl")

@gen function g()
    slope ~ normal(0, 1)
    return slope
end

np = pyimport("numpy")
# print(np.linspace(2000,2020,1000))
t=np.linspace(2000,2020,10)
head=(t.-t[1])*(-1)+5*np.cos(np.pi*2*t)

trace = Gen.simulate(geo_model, (head,t))

choices = Gen.get_choices(trace)

print("SIMULATION COMPLETED")
print(choices[:Kv])
print(trace[:Kv])

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