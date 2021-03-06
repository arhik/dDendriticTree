# Simple Tree
using MacroTools: @forward
using SparseArrays
using Random
using Flux
using Revise
using DataStructures
using SPLosses
using ZygoteRules: AContext, pullback
import Zygote: _pullback
import Zygote: @adjoint
import Flux: @functor
import Distributions: sample  # for synapse initialization optim

using MLDatasets
using Images

function test(x::Function; y::Int)
    x = 1 + y
    
    
end
#= ADDRESSED ISSUES
# weights are limited to maxSize. treeCount should be as factor but maxSize is used instead
=#

# TODO function rndActivateBitArray(count::Int, )::BitArray end
# TODO room for boosting [After succes of VQVAE it would be easier to jump on to this one]
# TODO define a custom weights structure [there is a temporary solution and it is mostly sufficient for now]
# TODO weights are managed in the forward pass need a better solution [same as previous problem]
# TODO stats for removing the bits and allocating new bits on the fly
# TODO better strategy to move the bit locations around 
# TODO make sofit shift code learning ...
# TODO competitive learning comparision and intuition for it paper on it.
# TODO MNIST autoencoder 
# TODO motion estimation after sofic shift estimation.
# TODO grid cells approach to it.
# TODO non binary equivalent ... obviously it might end up being graph neural networks.
# TODO connectivity is too vast. fixed using maxSize treeCount should be as factor but maxSize is used instead
# TODO more elegant version would be to take any dimension of input. Using Indices is a solution.



#= CURRENTLY BEING ADDRESSED

=#

traindata = float32.(MNIST.traintensor())

mutable struct Tree
    idxs::BitArray{1}
    weights::Array{Float32, 1}
    threshold::Float32
end

@functor Tree (weights,)

function _pullback(cx::AContext, t::Tree, x::BitArray)
    y = t(x)
    back =  (Δ) -> begin
        o = ones(Float32, size(t.weights))
        if Δ .== true
            o[y .== Δ] .= 1.0f0
        else
            o[y .== Δ] .= -1.0f0
        end
        ((idxs=nothing, weights=o, threshold=nothing), x) # TODO needs to think of nothing here
    end
    y, back
end

Tree(inputSize::Int; maxSize=100, threshold=0.6f0) = begin
    bits = bitrand(inputSize)
    returnbits = false.*BitArray(undef, size(bits))
    idxs = sample(findall(bits .== true), min(sum(bits), maxSize), replace = false)
    returnbits[idxs] .= true
    Tree(returnbits, rand(Float32, sum(returnbits)), threshold)
end

(t::Tree)(x) = (min.(max.(0.0, t.weights)) .> t.threshold) .& x[t.idxs]

Base.show(io::IO, t::Tree) = println(io, "Tree($(size(t.idxs)), threshold = $(t.threshold))")

mutable struct TreeLayer
    treeArray::Array{Tree, 1}
    idxs::BitArray{1}
    weights::Array{Float32, 1}
    f::Function
end

function topK(x, k=5)
    h = PriorityQueue(Base.Order.Reverse)
    o = false.&BitArray(undef, size(x))
    for i in 1:length(x)
        if length(h) < k
            push!(h, i=>x[i])
        elseif h.xs[end][2] < x[i]
            delete!(h, h.xs[end][1])
            push!(h, i=>x[i])
        end
    end
    for i in keys(h) # TODO threads for large scale values
        o[i] = true
    end
    return o
end

# TODO in this version TreeLayer should present other layers values to the input to

TreeLayer(inputSize::Int, treeCount::Int; fLayer=topK, threshold=0.6) = TreeLayer(
    [Tree(inputSize) for i in 1:treeCount],
    bitrand(treeCount),
    rand(treeCount),
    # topK TODO topK needs to be learnt too
    fLayer # Or regularizer should take care of it.
)

functor(::Type{<:TreeLayer}, c) = c.treeArray, ls -> TreeLayer(ls...)

TreeLayer(xs::Array{Tree}) = TreeLayer(xs, bitrand(length(xs)), rand(length(xs)))


@functor TreeLayer (treeArray,)


function _pullback(cx::AContext, tl::TreeLayer, x::BitArray)
    ylayer = tl(x)
    back = (Δ) -> begin
        grads = []
        for (i, t) in enumerate(tl[ylayer])
            y, _back = _pullback(cx, t, x)
            push!(grads, _back(Δ[i]))
        end
        return (grads, Δ)
    end
    return ylayer, back
end

@forward TreeLayer.treeArray Base.iterate, Base.getindex, Base.first, Base.last, Base.firstindex, Base.lastindex, Base.length, Base.push!, Base.pop!, Base.delete!, Base.getindex, Base.size

(tl::TreeLayer)(x) = begin
    output = zeros(size(tl))
    i = Threads.Atomic{Int}(1)
    Threads.@threads for (i, t) in collect(zip(1:reduce(*, size(tl)), tl))
        output[i] = sum(t(x))
    end
    tl.weights = output
    return tl.f(output)
end

Base.show(io::IO, tl::TreeLayer) = println("TreeLayer(treeArray = $(size(tl.treeArray)), idxs = $(size(tl.idxs)), weights = $(size(tl.weights)), f = $(tl.f))")

tl = TreeLayer(10000,100)

tl(bitrand(10000))

x = bitrand(784)
y = bitrand(784)

model = Chain(
    TreeLayer(28*28, 1000),
    TreeLayer(1000, 1000),
    TreeLayer(1000, 28*28)
)

gs = gradient(() -> Flux.mse(model(x), y), Flux.params(model))

lossF(x, y) = sum(x .& y)

@adjoint lossF(x, y) = lossF(x, y), (Δ) -> (x .& y, y .& x )

gs = gradient((m) -> lossF(model(x), y), model)

opt = Descent(0.1)

function customTraining(opt, n)
    local lossValue
    local meanLoss = 0.0
    ps = Flux.params(model)
    for i in 1:n
        idx = 1# rand(1:50000)
        ximg = reshape(traindata[:, :, idx], (28*28, ))
        x = ximg.>0.0
        gs = gradient(ps) do
            lossValue = Flux.mse(model(x), x)
            return lossValue
        end
        meanLoss = (meanLoss + lossValue)/2.0
        if i % 100 == 0
            @info "loss [$i]: $meanLoss"
        end
        Flux.update!(opt, ps, gs)
    end
end

customTraining(opt, 10000)

idx = 1
x = reshape(traindata[:, :, idx], (28,28))
ximg = colorview(Gray, x.>0)

dec = model(x.>0)
yimg = colorview(Gray, reshape(dec, (28, 28)))




