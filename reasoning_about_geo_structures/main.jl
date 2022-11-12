using PyCall
using Gen

pushfirst!(pyimport("sys")."path", "/mnt/reasoning_about_geo_structures/")
@pyinclude("/mnt/reasoning_about_geo_structures/inference.py")

function run(file_path::String, nr_iterations::Integer)
    py"perform_inf"(file_path, nr_iterations)
end