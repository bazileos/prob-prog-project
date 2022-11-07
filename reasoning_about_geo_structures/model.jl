include("./deformation.jl")

using PyCall
using Distributions
using Gen


@pyimport numpy

@gen function geo_model(head, reference_time)
    Kv ~ cauchy(-5, 3)# m/yr
    Sskv ~ cauchy(-3.5, 3) #m-1
    Sske ~ cauchy(-5, 3) #m-1
    nclay ~ uniform_discrete(5,10)
    claythick=5 # m

    # run simulation
    t,defm,head,defm_v=py"calc_deformation"(reference_time,head,10^Kv,10^Sskv,10^Sske,claythick,nclay)
    aligned_deformation=numpy.interp(reference_time,t,defm)

    # The following loop represents this method:
    # observe(observed_deformation, Normal(aligned_deformation, 2))
    for (index, value) in enumerate(aligned_deformation)
        {(:obs, index)} ~ normal(value, 2)
    end
    return
end 