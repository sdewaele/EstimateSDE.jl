# Maximum likelihood estimation from a given initial point
using EstimateSDE
using Random
using Printf
using Distributions

## Settings
proc = testprocess(:C)
N = 200
Tav = 0.5
Tkld = 0.5
engine = SdeRoots

data = rand(MersenneTwister(12202),proc,N,Tav)
problem = ProblemLike(data;p=EstimateSDE.order(proc))
tor = MlEstimator()
procstart = transform_ss(problem,proc)
proc_ml,opt = ml_estimate_init(:dynamichmc,tor,problem,procstart)

D = kldiv(proc,proc_ml,(0:Tkld:N*Tav))

## Report
printstyled("Actual:\n",color=:blue)
describe_model(proc)
printstyled("Estimate:\n",color=:blue)
describe_model(proc_ml)
@printf "KL divergence: %0.1f\n" D
