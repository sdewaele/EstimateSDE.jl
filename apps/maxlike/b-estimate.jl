# Maximum likelihood estimation
using EstimateSDE
using Random
using Printf
using Distributions
using Parameters

## Settings
proc = testprocess(:C)
N = 200
Tav = 0.5
Tkld = 0.5
tor = MlEstimator()
rng = MersenneTwister(12202)

data = rand(rng,proc,N,Tav)
problem = ProblemLike(data;p=EstimateSDE.order(proc),engine=ElimQ)

Ts = [0.5,1.0,2.0]*Tkld
init_tors = default_init_tors(rng,Ts)
proc_ml = estimate(tor,problem;init_tors=init_tors,throw_errors=true)

D = kldiv(proc,proc_ml,(0:Tkld:N*Tav))

## Report
printstyled("Actual:\n",color=:blue)
describe_model(proc)
printstyled("Estimate:\n",color=:blue)
describe_model(proc_ml)
@printf "KL divergence: %0.1f\n" D