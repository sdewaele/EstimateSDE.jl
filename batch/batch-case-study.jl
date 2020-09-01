## Case study from batch run
ENV["GKS_ENCODING"] = "utf-8"
using EstimateSDE
using Serialization
using Pipe: @pipe
using Plots
using DataFrames
using Statistics
using StatsPlots
using DataStructures

## Load settings and aggregated results
results_dir = settings_resultsdir((name="experiment01",))
S = deserialize(joinpath(results_dir,"settings.jld"))
df_eval = deserialize(joinpath(results_dir,"model-evaluation.jld")).df
df_data = deserialize(joinpath(results_dir,"data.jld")).df
df_eval = innerjoin(df_eval,df_data,on=:index)

df_kld_mean = deserialize(joinpath(results_dir,"aggregate.jld")).df

report_dir = joinpath(results_dir,"results")

## Select case
df = filter(x->x.proclabel==:ICMLA_C&&x.method=="MLE",df_eval)
index = df.index[end]
df = filter(x->x.index==index,df)
display(select(df,[:proclabel,:N,:engine,:kld]))

## Data
row_data = filter(x->x.index==index,df_data) |> first
data = row_data.data
t,y = (getfield.(data,x) for x in (:t,:y))
plot(t,y,ms=2,markershape=:square,xlabel="t",ylabel="y",legend=false)
plot!(title=string("Process ",row_data.proclabel," Tav = ","N = ",length(data)))

## Plot traces
Tkld = S.mle_trace.Tkld
proc = testprocess(row_data.proclabel)
duration = row_data.N*row_data.Tav
t_kld = (0:Tkld:duration)

traces = map(eachrow(df)) do x
  x.engine=>x.mle_trace
end |> OrderedDict

for (k,trace) in traces
  plts = reporttrace(data,trace,Tkld;proc=proc,t_kld=t_kld)
  plot!(plts.fit_order_wide,title=k)
end

## Estimation rerun
# ENV["JULIA_DEBUG"] = EstimateSDE
# r = filter(x->x.engine==ElimQ,df) |> first
# S.mle_trace
# spec = (engine=r.engine,init_actual=r.init_actual)
# trace = batch_mle_trace_single(S.mle_trace,spec,row_data,throw_errors=false)
# plts = reporttrace(data,trace,Tkld,proc=proc,t_kld=t_kld)
# plot!(plts.fit_order_wide,title="ElimQ rerun")
# nothing