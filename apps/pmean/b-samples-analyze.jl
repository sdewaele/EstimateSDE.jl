# Analyze samples generated in pmean/a-samples.jl
using EstimateSDE
using DynamicHMC
using DynamicHMC.Diagnostics
using Parameters
using Statistics
using DataStructures
using Printf
using MCMCDiagnostics
using Serialization
using Plots
using StatsBase
using Random

## Functions
format_plot!() = plot!(yaxis=:log,xlabel="f",ylabel="h",title="Power spectrum",
  fg_legend=:transparent, bg_legend=:transparent)

## Analysis settings
nspec = 1000

## Load settings, data and results
d = EstimateSDE.resultsdir("a-samples")
fn = joinpath(d,"samples.jld")
res = deserialize(fn)
@unpack settings,proc_ml,problem,procm,procs,R = res
@unpack proc,N,Tav,Tkld = settings

## Reported processes
pr = OrderedDict(
  "actual"=>proc,
  "ML"=>proc_ml,
  "PMean"=>procm,
)

## Sampling
for (i,r) in enumerate(R)
  println("Chain $i")
  println(summarize_tree_statistics(r.inference.tree_statistics))
end

## Rhat
𝓣 = transformation(problem,proc_ml)
θss = [𝓣.(r.inference.chain) for r in R]
R̂ = potential_scale_reduction(θss...)
println("R̂:")
println.("$n : $v" for (n,v) in pairs(R̂))

## Errors
tkld = 0:Tkld:N*Tav
D = OrderedDict(k=>kldiv(proc,v,tkld) for (k,v) in pr)

## Power spectrum
fkld = 1/Tkld/2
f = (0:nspec-1)/nspec*fkld
H = OrderedDict(k=>powerspectrum(proc.eq,f) for (k,proc) in pr)

## Plot results
P = plot()
for (k,h) in H
  lw = k=="actual" ? 2 : 1
  label = k
  if label≠"actual"
    label *= @sprintf " % 4.1f" D[k]
  end
  plot!(f,h,linewidth=lw,label=label)
end
format_plot!()
display(P)

## Posterior power spectrum
S = length(procs)
hall = Matrix{Float64}(undef,S,nspec)
for (i,proc) in enumerate(procs)
  hall[i,:] = powerspectrum(proc.eq,f)
end
prob_quantile = [0.1,0.5,0.9]
hqs = Matrix{Float64}(undef,nspec,length(prob_quantile))
for (i,x) in enumerate(eachcol(hall))
  hqs[i,:] = quantile(x,prob_quantile)
end
hmedian = hqs[:,2]
hlow = hqs[:,begin]
hhigh = hqs[:,end]

## Plot posterior power spectrum
# Method 1: uncertainty interval
P = plot(f,H["actual"],linewidth=2,label="actual")
plot!(f,hmedian,ribbon=(hlow,hhigh),fillalpha=0.1,label="posterior")
format_plot!(); display(P)

## Method 2: sample spectra
Splot = 20
ii = sample(MersenneTwister(785478),1:S,Splot;replace=false,ordered=true)
P = plot(f,H["actual"],linewidth=2,label="actual")
isfirst = true
for h in eachrow(hall[ii,:])
  global isfirst
  label = isfirst ? "posterior ($Splot)" : false
  plot!(f,h,color=:red,alpha=0.3,label=label)
  isfirst = false
end
format_plot!(); display(P)
