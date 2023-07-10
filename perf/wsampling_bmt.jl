# Benchmark on weighted sampling

using BenchmarkTools
using StatsBase

import StatsBase: direct_sample!, alias_sample!

import WeightedSampling: sample_heap!, WeightedSampler


### procedure definition

mutable struct WSampleProc{Alg} end

abstract type WithRep end
abstract type NoRep end

mutable struct Direct <: WithRep end
tsample!(s::Direct, wv, x) = direct_sample!(1:length(wv), wv, x)

mutable struct Alias <: WithRep end
tsample!(s::Alias, wv, x) = alias_sample!(1:length(wv), wv, x)

mutable struct Direct_S <: WithRep end
tsample!(s::Direct_S, wv, x) = sort!(direct_sample!(1:length(wv), wv, x))

mutable struct Sample_WRep <: WithRep end
tsample!(s::Sample_WRep, wv, x) = sample!(1:length(wv), wv, x; ordered=false)

mutable struct Sample_WRep_Ord <: WithRep end
tsample!(s::Sample_WRep_Ord, wv, x) = sample!(1:length(wv), wv, x; ordered=true)

mutable struct NoRep_Heap <: NoRep end
tsample!(s::NoRep_Heap, wv, x) = sample_heap!(WeightedSampler(wv.values), x; ordered=false, replace=true)


# config is in the form of (n, k)

Base.string(p::WSampleProc{Alg}) where {Alg} = lowercase(string(Alg))

Base.length(p::WSampleProc, cfg::Tuple{Int,Int}) = cfg[2]
Base.isvalid(p::WSampleProc{<:WithRep}, cfg::Tuple{Int,Int}) = ((n, k) = cfg; n >= 1 && k >= 1)
Base.isvalid(p::WSampleProc{<:NoRep}, cfg::Tuple{Int,Int}) = ((n, k) = cfg; n >= k >= 1)

function start(p::WSampleProc, cfg::Tuple{Int,Int})
    n, k = cfg
    x = Vector{Int}(undef, k)
    w = weights(fill(1.0/n, n))
    return (w, x)
end

Base.run(::WSampleProc{Alg}, cfg::Tuple{Int,Int}, s) where {Alg} = tsample!(Alg(), s[1], s[2])
# Base.done(p::WSampleProc, cfg, s) = nothing


### benchmarking

## benchmark group definition

bench_group = BenchmarkTools.BenchmarkGroup(
    "WithRep" => BenchmarkGroup(),
    "WithoutRep" => BenchmarkGroup())

ns = 5 * (2 .^ 0:9)
# ns = [8,16]
# ks = [1,4,8,16]
ks = 2 .^ 1:16

## with replacement

procs1 = [
                WSampleProc{Direct}(),
                WSampleProc{Alias}(),
                WSampleProc{Sample_WRep}(),
                WSampleProc{Direct_S}(),
                WSampleProc{Sample_WRep_Ord}(),
                WSampleProc{NoRep_Heap}()
               ]

cfgs1 = vec([(n, k) for k in ks, n in ns])

bg_wr = bench_group["WithRep"]
for cfg in cfgs1, p in procs1
    if !isvalid(p, cfg)
        continue
    end
    n, k = cfg
    !haskey(bg_wr, n) && (bg_wr[n] = BenchmarkGroup(["n=$n"]))
    !haskey(bg_wr[n], k) && (bg_wr[n][k] = BenchmarkGroup(["k=$k"]))
    !haskey(bg_wr[n][k], p) && (bg_wr[n][k][p] = BenchmarkGroup([string(p)]))

    bg_wr[n][k][string(p)] = @benchmarkable run($p, $cfg, s) setup=(s=start($p, $cfg)) samples=10_000 evals=1000
end

###

using DataFrames
import DataFrames: DataFrame

import BenchmarkTools: BenchmarkGroup, Trial

levels = [:replacement, :n, :k, :proc]

function descend(label, B::BenchmarkGroup, levels, i=1, nt=(;))
    level = levels[i]
    reduce(vcat, ( 
        begin
            if !isnothing(label)
                new_nt = NamedTuple{(keys(nt)...,level)}((values(nt)...,label))
            else
                new_nt = nt
                i = 0
            end
            descend(key, value, levels, i+1, new_nt)
        end
        for (key, value) in B
    ), init=[])
end

descend(label, T::Trial, level, i, nt) = NamedTuple{(keys(nt)...,:proc,:trial)}((values(nt)...,label, T))
      
descend(B::BenchmarkGroup, levels) = descend(nothing, B, levels)

DataFrames.DataFrame(B::BenchmarkGroup, levels::Vector{Symbol}) = DataFrame(descend(B, levels))


# rtable1 = run(procs1, cfgs1; duration=0.2)
# println()


# ## show results

# println("Sampling With Replacement")
# println("===================================")
# show(rtable1; unit=:mps, cfghead="(n, k)")
# println()
