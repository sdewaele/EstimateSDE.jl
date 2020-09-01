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
problem = ProblemLike(data;p=EstimateSDE.order(proc))

Ts = [0.5,1.0,2.0]*Tkld
init_tors = default_init_tors(rng,Ts)

mle1 = estimate(tor,lowerorder(problem,1);init_tors=init_tors)
m = init_tors[4]
initial_estimate(m,lowerorder(problem,2),[mle1])

##
proc = testprocess(:C)
problem = lowerorder(problem,order(proc))

r = combineroots([-1200.0,],[-800-200im])
proc = SdeProcess(SdeRoots(r,1.0))
procin = bringin(problem,proc)
