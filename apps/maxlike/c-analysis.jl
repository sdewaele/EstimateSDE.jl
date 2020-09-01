# Maximum Likelihood estimate with result analysis

using EstimateSDE
using Random
using Distributions
using Printf
using DataStructures
using Plots

## Settings
proc = testprocess(:ICMLA_B)
N = 200
Tav = 1.0
Tkld = 0.5 #0.2
p = 4
nspec = 1000
init_from_actual = false

## Start main
models = OrderedDict()
models["actual"] = proc
printstyled("Actual:\n",color=:blue)
describe_model(proc)

## Data
data = rand(MersenneTwister(5646356),proc,N,Tav)
problem = ProblemLike(data;p=p)

## Maximum Likelihood estimate
tor = MlEstimator()
Ts = [0.5,1,2]*Tkld
rng = MersenneTwister(35723)
init_tors = default_init_tors(rng,Ts)

if init_from_actual
  global proc
  push!(init_tors,InitialModel(proc)); @info "Adding actual as initial model"
end

## Maximum Likelihood estimate
mle,trace = @time estimate(tor,problem;init_tors=init_tors,ret_trace=true,throw_errors=false)

models["ML"] = mle
t_kld = (0:Tkld:N*Tav)
D_ml = kldiv(proc,mle,t_kld)
println("ML estimate: $mle\nKLD : $D_ml")

## Report
# Data
t = [x.t for x in data]; y = [x.y for x in data]
plt_data = plot(t,y,xlabel="t",ylabel="y",markershape=:x,fg_legend=nothing,bg_legend=nothing)
display(plt_data)
# Trace
plts = reporttrace(data,trace,Tkld;proc=proc,t_kld=t_kld)
nothing
