## Aggregate batch results
using EstimateSDE
using Serialization
using Pipe: @pipe
using DataFrames

# name = "experiment01"

## Load batch settings and results
results_dir = settings_resultsdir(name)
S = deserialize(joinpath(results_dir,"settings.jld"))
df_eval = deserialize(joinpath(results_dir,"model-evaluation.jld")).df
df_data = deserialize(joinpath(results_dir,"data.jld")).df

df_eval = @pipe(
  innerjoin(df_eval,df_data,on=:index) |>
  rename(_,:label=>:kld_label,:T=>:kld_T))

## Aggregate
df_kld_mean = @pipe(
  groupby(df_eval,[:proclabel,:engine,:init_actual,:method,:pmethod,:kld_label,:kld_T]) |>
  combine(_) do df
    x = df.kld
    c = length(x)
    μ = mean(x); σ = std(x)/√c
    (kld=μ,kld_std=σ,count=c)
  end
)

## Save to file
f = joinpath(results_dir,"aggregate.jld")
R = (settings=nothing,df=df_kld_mean,status=systemstatus())
serialize(f,R)
printstyled("\nSaved to file:\n$f",color=:blue,bold=true)
