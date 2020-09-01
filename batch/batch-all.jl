## All steps of batch processing 
# Data from previous steps is always read from file, so any step can
# be run separately from previous steps
using EstimateSDE
using Random
using Serialization
using Base.Threads
using Pipe: @pipe
using DataFrames

## Experiment name —————————————
name = "experiment02"

## Generate settings —————————————
S = (
  data=BatchDataSettings(;
    rng=MersenneTwister(9845789),
    name=name,
    # proclabels=[:ICMLA_W,:ICMLA_A,:ICMLA_B,:ICMLA_C],
    proclabels=[:ICMLA_A,:ICMLA_B,:ICMLA_C],
    Tav=1.0,
    Tmin=0.01,
    Ns=[500],
    Nrun=25
  ),
  mle_trace=BatchMlEstimation(;
    p=8,
    Tkld=0.2,
    engines=[ElimQ,SdeRootsEngine],
    init_actuals=[false]
  )
)

results_dir = settings_resultsdir(name)
f = joinpath(results_dir,"settings.jld")
serialize(f,S); @info "Settings saved to $f"

## Load settings —————————————
results_dir = settings_resultsdir(name)
S = deserialize(joinpath(results_dir,"settings.jld"))

## Thread check —————————————
nt = nthreads()
if nt==1
  @warn "nthreads==1. Increase number of threads for faster processing"
else
  @info "Threads used: $nt"
end

## Data —————————————
printstyled("Data set generation settings:\n",color=:blue)
println(S.data)
output_data = @time batch_data(S.data)
nothing

## Maximum Likelihood trace —————————————
# Settings
printstyled("\nSettings MLE trace\n",color=:blue)
print(S.mle_trace)

df_data = deserialize(joinpath(results_dir,"data.jld")).df

@info "Maximum Likelihood trace"
output_mle_trace = @time batch_mle_trace(S.mle_trace,df_data,results_dir)

## Posterior samples —————————————
# Skipped

## Derived estimates —————————————
df_mle_trace = deserialize(joinpath(results_dir,"mle-trace.jld")).df

@info "Maximum Likelihood derived estimates"
df_deriv = derived_estimates(df_mle_trace,results_dir)

## Model evaluation —————————————
df_deriv = deserialize(joinpath(results_dir,"derived-estimates.jld")).df
df_data = deserialize(joinpath(results_dir,"data.jld")).df

@info "Model evaluation"
output_eval = @time batch_model_evaluation(df_data,S.mle_trace,df_deriv,results_dir)

## Aggregate —————————————
@info "Aggregation"
include("aggregate.jl")

## Report —————————————
@info "Report"
include("figures.jl")