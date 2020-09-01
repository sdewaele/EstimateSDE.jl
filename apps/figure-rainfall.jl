# Analyze Monsoon rainfail data from
# @Article{sinha2015trends,
#   Title                    = {Trends and oscillations in the Indian summer monsoon rainfall over the last two millennia},
#   Author                   = {Sinha, A. and Kathayat, G. and Cheng, H. and others},
#   Journal                  = {Nature communications},
#   Year                     = {2015},
#   Volume                   = {6},
#   Publisher                = {Nature Publishing Group},
# }
using EstimateSDE
using Plots
using DataFrames
using Pipe:@pipe
pgfplotsx()
using Serialization
using Parameters
using Printf

## Load results rainfall-estimate
results_dir = EstimateSDE.resultsdir("rainfall-data")
f = joinpath(results_dir,"rainfall-sde-estimate.jld")
R = deserialize(f); @info "Results loaded to: $f"

## Settings
nspec = 500 # Number of points in the spectrum
ssize = 300,200

@unpack models,settings = R
@unpack Tkld,Tsub = settings

colors = [:blue,:red,:green,:orange,:yellow]

## Report
fmax = 1/2Tkld
f = (0:nspec-1)/nspec*1/2Tkld
plt_psd = plot(xlabel="f [1/year]",ylabel="psd",size=ssize,fg_legend=nothing,bg_legend=nothing)
for (i,m) in enumerate(models[1:end-1])
  h = powerspectrum(m.proc,f)
  kwargs = if m.label=="original"
    (color=:black,linestyle=:dash,label="all data")
  else
    (linealpha=0.8,label=nothing,color=colors[i-1])
  end
  plot!(f,h,label=m.label,yscale=:log10;kwargs...)
  println(m.label)
  describe_model(m.proc)
end
fmax_sub = 1/2Tsub
plot!([fmax_sub],linestyle=:dash,linecolor=:black,seriestype=:vline,label=nothing,
    legend=:topright)
# display(plt_psd) # fails in vscode with linealpha problem
@info "PSD plot not displayed, because it results in error in VS Code"

## trace plot
t_kld = 0:Tkld:100
proc = models[1].proc
plt_order = plot(xlabel="order p",ylabel="-L",fg_legend=nothing,bg_legend=nothing,size=ssize,
    legend=:topright)
for (i,m) in enumerate(models)
  if m.label=="original" continue end
  trace = m.trace
  data = m.data
  R,Href = tracetable(data,trace,proc,t_kld)

  # Maximum Likelihood per order
  mlo = @pipe(
    groupby(R,:p) |>
    combine(_) do df
      df[argmax(df.L),:]
    end |>
    sort(_,:p)
  )
  mlo_cols = [:p,:Hhat,:D,:init_tor_type,:stage_str]
  display(mlo[:,mlo_cols])
  # Overall maximum likelihood
  Imle = R[argmax(R.L),:]

  ## Plot trace
  Hhat_min = minimum(mlo.Hhat)
  Hhat = map(x->isinf(x) ? missing : x,mlo.Hhat)
  plot!(mlo.p,Hhat.-Hhat_min,ms=2,msc=:auto,markershape=:rect,label=nothing,
    color=colors[i-1],linealpha=0.8)
  plot!([Imle.p],[Imle.Hhat-Hhat_min],lc=nothing,mc=:yellow,markershape=:diamond,ma=0.5,ms=8,
      label=i==2 ? "ML" : nothing)
end
# @info "Plot fit vs order not shown"
plot!(ylim=(0,8))
# display(plt_order)

plt = plot(plt_psd,plt_order,layout=(2,1),size=(ssize[1],2ssize[2]))

## Save figure
for ext in ["pdf","tex"]
  f = joinpath(results_dir,"rainfall-results.$ext")
  savefig(plt,f)
  @info "Figure saved to $f"
end
