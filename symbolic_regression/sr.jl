using Gen
using Gen: get_retval
using Plots
using StatsBase

abstract type Node end
abstract type LeafNode <: Node end
abstract type BinaryOpNode <: Node end

"""
Constant
"""
struct Constant <: LeafNode
    param::Float64
end

function eval_ae(node::Constant, xs::Vector{Float64})
    n = length(xs)
    fill(node.param, n)
end

"""
Identify
"""
struct Identity <: LeafNode end

function eval_ae(node::Identity, xs::Vector{Float64})
    xs
end

"""
Plus node
"""
struct Plus <: BinaryOpNode
    left::Node
    right::Node
end

Plus(left, right) = Plus(left, right)

function eval_ae(node::Plus, xs::Vector{Float64})
    eval_ae(node.left, xs) .+ eval_ae(node.right, xs)
end

"""
Minus node
"""
struct Minus <: BinaryOpNode
    left::Node
    right::Node
end

Minus(left, right) = Minus(left, right)

function eval_ae(node::Minus, xs::Vector{Float64})
    eval_ae(node.left, xs) .- eval_ae(node.right, xs)
end

"""
Multiply node
"""
struct Multiply <: BinaryOpNode
    left::Node
    right::Node
end

Multiply(left, right) = Multiply(left, right)

function eval_ae(node::Multiply, xs::Vector{Float64})
    eval_ae(node.left, xs) .* eval_ae(node.right, xs)
end

"""
Divide node
"""
struct Divide <: BinaryOpNode
    left::Node
    right::Node
end

Divide(left, right) = Divide(left, right)

function eval_ae(node::Divide, xs::Vector{Float64})
    eval_ae(node.left, xs) ./ eval_ae(node.right, xs)
end

"""
Power node
"""
struct Power <: BinaryOpNode
    left::Node
    right::Node
end

Power(left, right) = Power(left, right)

function eval_ae(node::Power, xs::Vector{Float64})
    try
        eval_ae(node.left, xs) .^ eval_ae(node.right, xs)
    catch e
        if isa(e, DomainError)
            n = length(xs)
            fill(NaN, n)
        end
    end
end

Gen.get_retval(node::Constant) = node
Gen.get_retval(node::Identity) = node
Gen.get_retval(node::Plus) = node
Gen.get_retval(node::Minus) = node
Gen.get_retval(node::Multiply) = node
Gen.get_retval(node::Divide) = node
Gen.get_retval(node::Power) = node

function evaluate_expression(fn::Node, xs)
    eval_ae(fn, xs)
end

const CONSTANT = 1
const IDENTITY = 2
const PLUS = 3
const MINUS = 4
const MULTIPLY = 5
const DIVIDE = 6
const POWER = 7

const max_branch = 2

@gen function ae_prior(depth = 1)
    if depth > 5
        node_type = @trace(uniform_discrete(1, 2), :type)
    else
        node_type = @trace(uniform_discrete(1, 7), :type)
    end

    # constant node
    if node_type == CONSTANT
        param = @trace(uniform_continuous(0, 10), :param)
        node = Constant(param)

    # identity node
    elseif node_type == IDENTITY
        param = @trace(uniform_continuous(0, 10), :identity)
        node = Identity()

    # plus operator
    elseif node_type == PLUS
        left = @trace(ae_prior(depth+1), :left)
        right = @trace(ae_prior(depth+1), :right)
        node = Plus(left, right)

    # minus operator
    elseif node_type == MINUS
        left = @trace(ae_prior(depth+1), :left)
        right = @trace(ae_prior(depth+1), :right)
        node = Minus(left, right)

    # multiply operator
    elseif node_type == MULTIPLY
        left = @trace(ae_prior(depth+1), :left)
        right = @trace(ae_prior(depth+1), :right)
        node = Multiply(left, right)

    # divide operator
    elseif node_type == DIVIDE
        left = @trace(ae_prior(depth+1), :left)
        right = @trace(ae_prior(depth+1), :right)
        node = Divide(left, right)

    # power operator
    elseif node_type == POWER
        left = @trace(ae_prior(depth+1), :left)
        right = @trace(ae_prior(depth+1), :right)
        node = Power(left, right)

    # unknown node type
    else
        error("Unknown node type: $node_type")
    end

    return node
end

@gen function model(xs::Vector{Float64})
    n = length(xs)

    # sample arithmetic expression
    fn::Node = @trace(ae_prior(), :tree)    

    # evaluate expression
    ys = evaluate_expression(fn, xs)
    
    for i = 1:n
        @trace(normal(ys[i], 0.1), "y-$i")
    end
    
    return fn
end

@gen function pick_random_node_path(node::Node, path::Vector{Symbol})
    if isa(node, LeafNode)
        @trace(bernoulli(1), :done)
        path
    elseif @trace(bernoulli(0.5), :done)
        path
    elseif @trace(bernoulli(0.5), :recurse_left)
        push!(path, :left)
        @trace(pick_random_node_path(node.left, path), :left)
    else
        push!(path, :right)
        @trace(pick_random_node_path(node.right, path), :right)
    end
end

@gen function subtree_proposal(prev_trace)
    prev_subtree_node::Node = get_retval(prev_trace)
    (path::Vector{Symbol}) = @trace(pick_random_node_path(prev_subtree_node, Symbol[]), :choose_subtree_root)
    new_subtree_node::Node = @trace(ae_prior(), :subtree) # mixed discrete / continuous
    (path, new_subtree_node)
end

@transform subtree_involution_tree_transform (model_in, aux_in) to (model_out, aux_out) begin

    (path::Vector{Symbol}, new_subtree_node) = @read(aux_in[], :discrete)

    # populate backward assignment with choice of root
    @copy(aux_in[:choose_subtree_root], aux_out[:choose_subtree_root])

    # swap subtrees
    model_subtree_addr = isempty(path) ? :tree : (:tree => foldr(=>, path))
    @copy(aux_in[:subtree], model_out[model_subtree_addr])
    @copy(model_in[model_subtree_addr], aux_out[:subtree])
end

is_involution!(subtree_involution_tree_transform)

replace_subtree_move(trace) = metropolis_hastings(
    trace, subtree_proposal, (), subtree_involution_tree_transform; check=true)[1]

function inference(xs::Vector{Float64}, ys::Vector{Float64}, num_iters::Int)

    # observed data
    constraints = choicemap()
    n = length(ys)
    for i = 1:n
        constraints["y-$i"] = ys[i]
    end

    # generate initial trace consistent with observed data
    (trace, _) = generate(model, (xs,), constraints)

    # do MCMC
    local fn::Node
    local noise::Float64
    for iter=1:num_iters

        fn = get_retval(trace)

        # do MH move on the subtree
        trace = replace_subtree_move(trace)
    end
    fn
end

function find_expressions(xs::Vector{Float64}, ys::Vector{Float64}, num_iter::Integer)
    solutions = Vector{Tuple{Node, Float64}}(undef, 10)
    for i = 1:10
        trace = inference(xs, ys, num_iter)
        fn = get_retval(trace)
        solutions[i] = (fn, StatsBase.msd(ys, evaluate_expression(fn, xs)))
    end
    sort!(solutions, by = s -> s[2])
    solutions[1:5]
end

function find_and_print_expressions(xs::Vector{Float64}, ys::Vector{Float64}, num_iter::Integer)
    solutions = find_expressions(xs, ys, num_iter)
    n = length(solutions)
    for i = 1:n
        println(solutions[i][1])
    end
    solutions
end

function plot_results(solutions::Vector{Tuple{Node, Float64}}, test_function::Function)
    xs_plot::Vector{Float64} = 0.1:0.01:10
    ys_original = test_function(xs_plot)

    n = length(solutions)
    y_outputs = Vector{Vector{Float64}}(undef, n+1)
    labels = Vector{String}(undef, n+1)

    y_outputs[1] = ys_original
    labels[1] = "test fn"
    for i = 1:n
        y_outputs[i+1] = evaluate_expression(solutions[i][1], xs_plot)
        labels[i+1] = "#"*string(i)
    end

    plot(xs_plot, y_outputs, label = permutedims(labels))
end

function test_model(xs::Vector{Float64}, ys::Vector{Float64}, num_iter::Integer, test_fn::Function)
    solutions = find_and_print_expressions(xs, ys, num_iter)
    plot_results(solutions, test_fn)
end