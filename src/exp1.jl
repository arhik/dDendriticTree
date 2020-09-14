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

# TODO function rndActivateBitArray(count::Int, )::BitArray end

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

Tree(inputSize::Int; threshold=0.6f0) = begin
    bits = bitrand(inputSize) # Half of it is connected TODO
    Tree(bits, rand(Float32, sum(bits)), threshold)
end

(t::Tree)(x) = (t.weights .> t.threshold) .& x[t.idxs]

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


(tl::TreeLayer)(x) = begin
    output = zeros(size(tl))
    for (i, t) in enumerate(tl)
        output[i] = sum(t(x))
    end
    tl.weights = output
    return tl.f(output)
end



Base.show(io::IO, tl::TreeLayer) = println("TreeLayer(treeArray = $(size(tl.treeArray)), idxs = $(size(tl.idxs)), weights = $(size(tl.weights)), f = $(tl.f))")

tl = TreeLayer(10000,100)

tl(bitrand(10000))

x = bitrand(784)
y = bitrand(1000)

m = Chain(
    TreeLayer(28*28, 100),
    TreeLayer(100, 1000)
)

gs = gradient(() -> Flux.mse(m(x), y), params(m))

lossF(x, y) = sum(x .& y)

@adjoint lossF(x, y) = lossF(x, y), (Δ) -> (x .& y, y .& x )

gs = gradient((m) -> lossF(m(x), y), m)

# Base.convert(BitArray, y::Float32) = begin
#     bits = bitstring(y)
#     BitArray([x.=='1' for x in bits])
# end

#Base.convert(BitArray, y::Array{Float32}) = begin
#    bitarray = BitArray(undef, (sizeof(y)*8, size(y)...))
#    indices = CartesianIndices(size(y))
#    Threads.@thread for i in indices
#        bitarray[]
#    end
#end

# NAS is possible on the fly

