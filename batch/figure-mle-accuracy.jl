## Batch result figure: MLE accuracy and spectra
using EstimateSDE
using Serialization
using Pipe: @pipe
using Plots
pgfplotsx()
using DataFrames
using Statistics
using StatsPlots
using REPL.TerminalMenus
using Printf
using Random

## Settings
name = "experiment01"
Tkld = 0.2
nspec = 500 # Number of points in a power spectrum
npsd = 5 # Number of power spectra plotted
ssize = (300,300) # figure size

## Load settings and aggregated results
results_dir = settings_resultsdir(name)
S = deserialize(joinpath(results_dir,"settings.jld"))
R = deserialize(joinpath(results_dir,"aggregate.jld")).df
df_eval = deserialize(joinpath(results_dir,"model-evaluation.jld")).df
df_data = deserialize(joinpath(results_dir,"data.jld")).df
df_eval = @pipe(
  innerjoin(df_eval,df_data,on=:index) |>
  rename(_,:label=>:kld_label,:T=>:kld_T))

figure_dir = joinpath(results_dir,"figures")
if !isdir(figure_dir) mkpath(figure_dir) end

## Preprocessing for plotting
em = Dict(SdeRootsEngine=>"root",ElimQ=>"coef")
for dfp in [df_eval,R]
  dfp.procname = map(x->split(String(x),"_")[end],dfp.proclabel)
  dfp.param = [em[x] for x in dfp.engine]
end

## KLD for reference MLE
dfr = @pipe(
  filter(R) do x
    x.method=="MLE" && x.param=="root" &&
    x.procname∈["A","B","C"] &&
    x.kld_label=="regular" && x.kld_T==Tkld
  end |>
  select(_,[:procname,:kld,:kld_std])
)

klds = Dict(zip(dfr.procname,dfr.kld))

## Plot relative KLD vs process
plt_kld = plot(xlabel="process",ylabel="KLD",size=ssize)
plot!(seriestype=:bar,dfr.procname,dfr.kld,linecolor=:black,bar_width=0.5,yerror=dfr.kld_std,
  label=nothing)
plot!(legend=:none, size=(350,300),xlabel="process",ylabel="KLD")

## Power spectra for representative simulation runs
plts_psd = []
for procname in ["A","B","C"]
  # Select case
  dfr = @pipe(
    filter(df_eval) do x
      x.procname==procname && 
      x.method=="MLE" && x.param=="root" &&
      x.kld_label=="regular" && x.kld_T==Tkld
    end |>
    select(_,[:proclabel,:procname,:data,:N,:Tav,:est,:kld_T,:kld])
  )

  # Plot actual psd
  rr = first(dfr)
  proc = testprocess(rr.proclabel)
  duration = rr.N*rr.Tav
  t_kld = 0:rr.kld_T:duration
  f = (0:nspec-1)./nspec/2Tkld
  h = powerspectrum(proc,f)
  plt = plot(f,h,label="process $(rr.procname)",linewidth=2)
  plot!(yscale=:log10,xlabel="f",ylabel="psd",size=ssize,
      fg_legend=nothing,bg_legend=nothing,legend=:topright)
  push!(plts_psd,plt)

  # Plot `npsd` power spectra
  rng = MersenneTwister(234230)
  iir = rand(rng,1:nrow(dfr),npsd)
  for i in 1:npsd
    row_eval = dfr[iir[i],:]
    ĥ = powerspectrum(row_eval.est,f)
    lab = i==1 ? @sprintf("MLE D %0.0f",klds[procname]) : nothing
    plot!(f,ĥ,label=lab,color=:orange,linealpha=0.8)
  end
  fn = 1/2S.data.Tav
  vline!([fn],linecolor=:black,linestyle=:dash,label=nothing)
end

# Compbine into 2x2 figure
plt = plot(plt_kld,plts_psd...;layout=(2,2),size=2 .* ssize)
display(plt)

## Save
for ext in ["tex","pdf"]
  f = "mle-accuracy.$ext"
  savefig(plt,joinpath(figure_dir,f))
  @info "Saved $f"
end