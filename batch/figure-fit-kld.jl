## Figure of fit and kld vs model order
ENV["GKS_ENCODING"] = "utf-8"
using EstimateSDE
using Serialization
using Pipe: @pipe
using Plots
pgfplotsx()
using DataFrames
using Statistics
using StatsPlots
using DataStructures
using Printf

name="experiment01"
cmethods = ["MLE root","MLE coef","random init root"]
ssize = (300,200)

## Load settings and aggregated results
results_dir = settings_resultsdir(name)
S = deserialize(joinpath(results_dir,"settings.jld"))
df_data = deserialize(joinpath(results_dir,"data.jld")).df
df_deriv = deserialize(joinpath(results_dir,"derived-estimates.jld")).df
df_deriv = innerjoin(df_deriv,df_data,on=:index)
df_eval = deserialize(joinpath(results_dir,"model-evaluation.jld")).df
df_eval =  @pipe(
  innerjoin(df_eval,df_data,on=:index) |>
  rename(_,:label=>:kld_label,:T=>:kld_T))

figure_dir = joinpath(results_dir,"figures")

## Preprocessing for plotting
for dfc in [df_deriv,df_eval]
  dfc.procname = map(x->split(String(x),"_")[end],dfc.proclabel)
  em = Dict(SdeRootsEngine=>"root",ElimQ=>"coef")
  dfc.param = [em[x] for x in dfc.engine]
  mmap = Dict("MLE EXPAND"=>"expand","MLE init ALL_RANDOM"=>"random init")
  dfc.cmethod = map(eachrow(dfc)) do x
    m = get(mmap,x.method,x.method)
    m*" "*x.param
  end
end

## Select representative case
dfef = @pipe(
  filter(df_eval) do x
    x.proclabel==:ICMLA_C &&
    x.cmethod∈cmethods &&
    x.kld_label=="regular" && x.kld_T==0.2
  end
)

dfu = unstack(dfef,:index,:cmethod,:kld)

ii = (dfu[!,"MLE coef"].>1.5dfu[!,"MLE root"]) .& (dfu[!,"random init root"] .> 3dfu[!,"MLE root"])
dfus = dfu[ii,:]
# println("Datasets with varying KLD for different cmethods:")
# display(dfus)
index = dfus.index[2]
row_data = filter(r->r.index==index,df_data) |> first

## Plot traces
Tkld = S.mle_trace.Tkld
proc = testprocess(row_data.proclabel)
data = row_data.data
t_kld = 0:Tkld:row_data.N*row_data.Tav

## Start loop over cmethods
plt = plot(xlabel="order",ylabel="-L",fg_legend=nothing,bg_legend=nothing,size=ssize)
colors = Dict("MLE root"=>:orange,"MLE coef"=>:blue,"random init root"=>:green)
for cmethod in cmethods
  row_eval = filter(x->x.index==index&&x.cmethod==cmethod,dfef) |> first

  trace = row_eval.mle_trace
  if cmethod=="random init root"
    # MLE from random initiation
    trace = filter(trace) do x
      x.init_tor isa AddComplexRoot && x.init_tor.rootmode==EstimateSDE.ALL_RANDOM
    end  
  end

  ## Maximum likelihood per order
  R,Href = tracetable(data,trace,proc,t_kld)
  mlo = @pipe(
    groupby(R,:p) |>
    combine(_) do df
      df[argmax(df.L),:]
    end |>
    sort(_,:p)
  )
  println(cmethod)
  Imle = R[argmax(R.L),:]

  ## Plot
  col = colors[cmethod]
  plot!(mlo.p,mlo.Hhat,color=col,ms=2,markershape=:rect,msc=:auto,label=nothing)
  cmethod_plt = cmethod=="random init root" ? "random init" : cmethod
  # @printf "%s KLD = %0.0f\n" cmethod_plt Imle.D
  lab = cmethod_plt #@sprintf "%s [%0.0f]" cmethod_plt Imle.D
  plot!([Imle.p],[Imle.Hhat],lc=nothing,mc=col,markershape=:diamond,ma=0.3,ms=7,label=lab,
      legend=:bottomleft)
end
plot!(ylim=(-20,70))
display(plt)

## Save
for ext in ["tex","pdf"]
  f = "fit-kld.$ext"
  savefig(plt,joinpath(figure_dir,f))
  @info "Saved $f"
end