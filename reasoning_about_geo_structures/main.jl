using PyCall
using Gen

pushfirst!(pyimport("sys")."path", "./reasoning_about_geo_structures/")
@pyinclude("./reasoning_about_geo_structures/inference.py")

file_name = ARGS[1]
nr_iterations = ARGS[2]

py"perform_inf"(file_name, nr_iterations)

