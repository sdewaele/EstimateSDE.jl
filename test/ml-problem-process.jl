# Problem with ML ML estimator
# The initial point is valid;
# L-BFGS fails during line search for an SDE(2) model

using EstimateSDE
using EstimateSDE:AddSdeRoot,ResampleAr,InitialModel,estimate_trace,
  has_statstate, ml_estimate_init
using Random
using Distributions
using Printf
using DataStructures
using TypedTables
using StatsPlots
using JuliaDB
using .Threads

## Settings
proc0 = testprocess(:C)
proc = SdeProcess(proc0.eq) # Use stationary initial conditions
N = 200
Tav = 1.0
Tkld = 0.5
p = 2
Trs = [Tkld;Tav]
nspec = 1000

rng = MersenneTwister(539872437) # fails!
t,y = rand(rng,proc,N,Tav)
problem = SdeProblemLike(:stat,:elimQ,t,y,p,true,nothing)

## Maximum Likelihood estimate
tor = MlEstimator()
init_tors = []
push!(init_tors,AddSdeRoot(-10/Tav))
[push!(init_tors,ResampleAr(Tr,m,-10/Tav)) for Tr in Trs, m in [:nearest,:linear]]
est_ml,trace = estimate_trace(tor,problem,init_tors)

## Failed initiation
trace_failed = trace[3]
proc_ini = trace_failed[:proc_ini]
est,opt = ml_estimate_init(:dynamichmc,tor,problem,proc_ini)
