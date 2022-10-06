using Gen

struct BinaryOp
    fn::Function
    expr::String
end

@dist function uniform_draw_ops(ops::Vector{BinaryOp}) 
    i = uniform_discrete(0, 4)
    return ops[i]
end

plus = BinaryOp((a::Number, b::Number) -> a+b, "+")
minus = BinaryOp((a::Number, b::Number) -> a-b, "-")
multiply = BinaryOp((a::Number, b::Number) -> round(a*b), "*")
divide = BinaryOp((a::Number, b::Number) -> round(a/b), "/")
power = BinaryOp((a::Number, b::Number) -> a^b, "**")
identity = BinaryOp(x -> x, "x")

binaryOps = [plus, minus, multiply, divide, power]

@gen function randomConstantFunction()
    c ~ uniform_discrete(1, 10)
    return BinaryOp(x -> c, string(c))
end

@gen function randomCombination(f,g)
    op ~ uniform_draw_ops(binaryOps);
    opfn = op.fn
    ffn = f.fn
    gfn = g.fn
    return BinaryOp(x -> opfn(ffn(x),gfn(x)), f.expr+op.expr+g.expr)
end

# sample an arithmetic expression
@gen function randomArithmeticExpression()
    if bernoulli(0.5)
        return randomCombination(randomArithmeticExpression(), randomArithmeticExpression())
    else
        return bernoulli(0.5) ? identity : randomConstantFunction()
    end
end

@gen function symbolic_regression_model()
    randomArithmeticExpression()
end