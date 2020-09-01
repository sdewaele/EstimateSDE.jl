# Posterior samples and mean estimation from a given initial point
using EstimateSDE
using EstimateSDE: Parameters
using Serialization
using Random
using Parameters

## Settings
settings = (
  proc=testprocess(:B),
  N=100,
  Tav=1.0,
  Tkld=1.0,
)

@unpack proc,N,Tav,Tkld = settings
rng = MersenneTwister(12202)
data = rand(rng,proc,N,Tav)
problem = ProblemProb(data;p=EstimateSDE.order(proc),engine=ElimQ)

## Maximum likelihood estimate
mltor = MlEstimator()
init_tors = default_init_tors(rng,[0.5,1.0,2.0]*Tkld)
proc_ml = estimate(mltor,problem.like;init_tors=init_tors,throw_errors=false)
if proc_ml isa WhiteNoise
  error("No error; correct behavior: ML estimate is white noise; no posterior samples are taken.")
end

## Posterior samples
tor = PmeanEstimator(;rng=MersenneTwister(54858979),N=200,Nwarmup=200,nchains=4)
procm,procs,R = @time estimate(tor,problem,proc_ml)

## Save settings, data and results
d = EstimateSDE.resultsdir("a-samples")
fn = joinpath(d,"samples.jld")
res = Dict{Symbol,Any}()
@pack! res = settings,proc_ml,problem,procm,procs,R
serialize(fn,res)
@info "Settings and results saved to $fn"
