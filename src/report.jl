# Report: Plotting, display tables etc.

"
$SIGNATURES

Tables
* Properties ML per order

Plots
* KLD/Hhat against model order
* Power spectrum
* Autocorrelation
"
function reporttrace(data,trace,Tkld;proc=nothing,t_kld=nothing,
      ΔH_narrow=nothing,nspec=1000,dispfig=true)
  actual_given = !isnothing(proc)
  if !actual_given
    i = argmax(getfield.(trace,:L))
    proc = trace[i].est
  end
  if isnothing(t_kld)
    tstart = data[1].t; tend = data[end].t
    t_kld = tstart:Tkld:tend
  end
  if isnothing(ΔH_narrow)
    p = maximum(getfield.(trace,:p))
    ΔH_narrow = min(5,p)
  end
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
  println("Maximum Likelihood per order")
  display(select(mlo,mlo_cols))

  # Overall maximum likelihood
  Imle = R[argmax(R.L),:]
  println("Overall Maximum Likelihood")
  display(Imle[mlo_cols])

  println("Href = ",Href)

  # Fit/KLD against order
  plts = OrderedDict()
  for ylimtype in (:wide,:narrow)
    plt_order = plot(mlo.p,mlo.Hhat,color=:grey,markershape=:rect,linewidth=2,label="Hhat")
    lab = @sprintf "ML %i D %5.4g H %5.4g" EstimateSDE.order(Imle.est) Imle.D Imle.Hhat
    plot!(mlo.p,mlo.D,color=:red,linewidth=2,label="KLD ML",markershape=:rect)
    plot!([Imle.p],[Imle.Hhat],lc=nothing,mc=:yellow,markershape=:diamond,ma=0.5,ms=10,label=lab)
    plot!(xlabel="order",ylabel="Hhat / KLD",fg_legend=nothing,bg_legend=nothing)
    if ylimtype==:narrow
      plot!(title="small Hhat")
    end
    ymin = round(min(0,minimum(R.Hhat)),RoundDown;sigdigits=1)
    ymax = if ylimtype==:narrow
      ym = max(ΔH_narrow,minimum(R.Hhat))
      round(ym,RoundUp;sigdigits=1)
    else
      HM = maximum(filter(isfinite,mlo.Hhat))
      Dm = minimum(skipmissing(R.D))
      yh = round(max(HM,Dm),RoundUp;sigdigits=1)
      max(2ΔH_narrow,yh)
    end
    plot!(ylim=(ymin,ymax))
    plts[Symbol(:fit_order_,ylimtype)] = plt_order
  end

  # Models for plotting
  models = OrderedDict{String,TimeSeriesModel}("ML"=>Imle.est)
  if actual_given  models["actual"] = proc end

  # Power spectrum and autocorrelation
  f = (0:nspec-1)./nspec/2Tkld
  τ = 0:Tkld:(t_kld[end]-t_kld[1])
  plts[:psd] = plot(yscale=:log10,xlabel="f",ylabel="psd",title="power spectrum")
  plts[:acf] = plot(xlabel="t",ylabel="R",title="autocorrelation")
  for (k,m) in models
    h = powerspectrum(m,f)
    lab = k
    if k≠"actual" && actual_given
      D = kldiv(proc,m,t_kld)
      lab *=@sprintf " D %3.2g" D
    end
    plot!(plts[:psd],f,h,label=lab)
    plot!(fg_legend=:transparent,bg_legend=:transparent)
    R = autocov(m,τ)
    plot!(plts[:acf],τ,R,label=lab)
    plot!(fg_legend=:transparent,bg_legend=:transparent)
  end
  plt = plot(values(plts)...;layout=(2,2),size=(600,600))
  if dispfig  display(plt)  end
  return (;plts...)
end
