## Batch result figures: compare estimators
using EstimateSDE
using Serialization
using Pipe: @pipe
using Plots
pgfplotsx()
using DataFrames
using Statistics
using StatsPlots
using REPL.TerminalMenus

## Settings
name = "experiment01"
ssize = (300,220)

## Load settings and aggregated results
results_dir = settings_resultsdir(name)
S = deserialize(joinpath(results_dir,"settings.jld"))
R = deserialize(joinpath(results_dir,"aggregate.jld")).df

figure_dir = joinpath(results_dir,"figures")
if !isdir(figure_dir) mkpath(figure_dir) end

## Preprocessing for plotting
R.proclabel = string.(R.proclabel)
R.proclabel = map(x->split(String(x),"_")[end],R.proclabel)
em = Dict(SdeRootsEngine=>"root",ElimQ=>"coef")
R.param = [em[x] for x in R.engine]
mmap = Dict("MLE EXPAND"=>"expand","MLE init ALL_RANDOM"=>"random init")
R.cmethod = map(eachrow(R)) do x
  m = get(mmap,x.method,x.method)
  m*" "*x.param
end

## KLD of MLE by process and computation method
dfn = @pipe(
  filter(R) do x
    x.cmethod ∈ ["MLE root","MLE coef","random init root"] &&
    x.proclabel∈["A","B","C"] &&
    x.kld_label=="regular" && x.kld_T==0.2
  end |>
  select(_,[:proclabel,:cmethod,:kld,:kld_std]) |>
  groupby(_,[:proclabel]) |>
  combine(_) do df
    kld_norm = filter(x->x.cmethod=="MLE root",df).kld[1]
    dfc = copy(df)
    dfc[:,:kld] = df.kld./kld_norm
    dfc[:,:kld_std] = df.kld_std./kld_norm
    dfc
  end
)

## Bar plot of relative KLD per engine, per process label
plt = @df dfn groupedbar(:proclabel,:kld,group=:cmethod,bar_width=0.8,yerror=:kld_std,
  linecolor=:black) #,markerstrokecolor=:auto) #lw=0 removes line around box
plot!(legend=:topleft,fg_legend=nothing,bg_legend=nothing, size=ssize,
  xlabel="process",ylabel="relative KLD",ylim=(0,5))
display(plt)

## Save figure
for ext in ["tex","pdf"]
  f = "cmp-estimators.$ext"
  savefig(plt,joinpath(figure_dir,f))
  @info "Saved $f"
end
