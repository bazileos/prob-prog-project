include("./deformation.jl")

using PyCall
using Distributions
using Gen


@pyimport numpy

@gen function geo_model(reference_time, head)
    o = Float64[]
    try
        Kv ~ cauchy(-5, 3)# m/yr
        Sskv ~ cauchy(-3.5, 3) #m-1
        Sske ~ cauchy(-5, 3) #m-1
        nclay ~ uniform_discrete(5,10)
        claythick=5 # m

        t,defm,_,_=py"calc_deformation"(reference_time,head,10^Kv,10^Sskv,10^Sske,claythick,nclay)

        aligned_deformation=numpy.interp(reference_time,t,defm)
        print(aligned_deformation)
        # The following loop represents this method:
        # observe(observed_deformation, Normal(aligned_deformation, 2))
        for (index, value) in enumerate(aligned_deformation)
            push!(o, {(:obs, index)} ~ normal(value, 0.001))
            
        end
    catch _
        println("Error in calc_deformation")
    end
    o
end 


@gen function geo_model_dif(reference_time, head)
    @param Kv::Float32
    @param Sskv::Float32
    @param Sske::Float32
    n_clay = 7
    # Kv ~ cauchy(-5, 3)# m/yr
    # Sskv ~ cauchy(-3.5, 3) #m-1
    # Sske ~ cauchy(-5, 3) #m-1
    # nclay ~ uniform_discrete(5,10)
    claythick=5 # m

    # run simulation

    t,defm,_,_=py"calc_deformation"(reference_time,head,1/(10^-Kv),1/(10^-Sskv),1/(10^-Sske),claythick,n_clay)
    aligned_deformation=numpy.interp(reference_time,t,defm)

    # The following loop represents this method:
    # observe(observed_deformation, Normal(aligned_deformation, 2))
    for (index, value) in enumerate(aligned_deformation)
        {(:obs, index)} ~ normal(value, 2)
    end
    return
end 