using Gen

function perform_geo_inference(o::Vector{Float32}, num_iters::Int)
    


    choices = get_choices(trace)
    Kv = choices[:Kv]
    Sskv = choices[:Sskv]
    Sske = choices[:Sske]
    n_clay = choices[:nclay]
    return Kv, Sskv, Sske, n_clay