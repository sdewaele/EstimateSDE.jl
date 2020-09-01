## Kullback-Leibler Discrepancy as a function of the model order
# for white noise
using EstimateSDE
using Serialization
using Pipe: @pipe
using Plots
pgfplotsx()
using DataFrames
using Statistics
using StatsPlots
using REPL
using REPL.TerminalMenus

## Settings
name = "experiment01"
ssize = (300,200)

## Load batch settings and results
results_dir = settings_resultsdir(name)
S = deserialize(joinpath(results_dir,"settings.jld"))
df_eval = deserialize(joinpath(results_dir,"model-evaluation.jld")).df
df_data = deserialize(joinpath(results_dir,"data.jld")).df

df_eval = @pipe(
  innerjoin(df_eval,df_data,on=:index) |>
  rename(_,:label=>:kld_label,:T=>:kld_T))

## Generate graphs
figure_dir = joinpath(results_dir,"figures")
if !isdir(figure_dir) mkpath(figure_dir) end

@info "Figures are saved to $figure_dir"

## Count of selected model order
colfix = (engine=SdeRootsEngine,proclabel=:ICMLA_W,method="MLE pmax",kld_label="same")
# colfix = (engine=SdeRootsEngine,proclabel=:ICMLA_W,method="MLE pmax",kld_label="regular",kld_T=1.0)
df = @pipe(
  filter(df_eval) do x
    for (k,v) in pairs(colfix)
      if x[k]≠v return false end
    end
    return true
  end |>
  select(_,[:pmethod,:kld,:p])
)

## KLD as a function of model order
D = @pipe(
  groupby(df,:pmethod) |>
  combine(_) do df
    x = df.kld
    c = length(x)
    μ = mean(x); σ = std(x)/√c
    (kld=μ,kld_std=σ,count=c)
  end
)

## Theoretical result
pmax = maximum(D.pmethod)
ps = 0:pmax
kld_theory = 0.5*(ps.+1) # parameters: p SDE coefficients plus one for σϵ

## KLD vs order
plt = plot(xlabel="p",ylabel="Do",fg_legend=nothing,bg_legend=nothing,
  legend=:topleft,size=ssize,ylim=(0,5))
plot!(ps,kld_theory,linecolor=:gray,markercolor=:gray,markershape=:square,
  markerstrokecolor=:auto,markersize=2,label="theory")
plot!(D.pmethod,D.kld,yerror=D.kld_std,markerstrokecolor=:auto,label="SDE(p)",
  markershape=:square,markersize=2)
display(plt)

## Save figure
for ext in ["tex","pdf"]
  f = "whitenoise.$ext"
  savefig(plt,joinpath(figure_dir,f))
  @info "Saved $f"
end