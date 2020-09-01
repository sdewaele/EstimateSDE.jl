# Batch posterior samples
# Enable multiple threads and sysimage by starting julia using:
# JULIA_NUM_THREADS=10 julia --sysimage=/home/student/.julia/environments/v1.4/JuliaSysimage.so

using EstimateSDE
using Serialization
using Base.Threads

## Batch settings
f = joinpath(@__DIR__,"batch-settings.toml")
setttings_batch = batchsettings(f)

## Maximum likelihood results
dd = EstimateSDE.resultsdir("ICMLA",setttings_batch.name,string(setttings_batch.proclabel))
datafilename = joinpath(dd,"maximum-likelihood.jld")
println(datafilename)
output_mle = deserialize(datafilename)
@info "Read maximum likelihood results from $datafilename"

## Settings
settings = (
  tor_kwargs=(N=200,Nwarmup=200,nchains=1),
  Nest=1 # Nothing: do all
)
printstyled("\nSettings posterior samples\n",color=:blue)
println(settings)

## Batch posterior samples
ENV["JULIA_DEBUG"] = EstimateSDE
if nthreads()==1
  @warn "nthreads==1. Increase number of threads for faster processing"
end
output_posterior_samples = @time batch_posterior_samples(settings,output_mle)
nothing