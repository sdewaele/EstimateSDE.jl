# Batch simulations

"
$SIGNATURES

Settings for batch data generation
"
Base.@kwdef struct BatchDataSettings
  "Master Random Number Generator"
  rng::MersenneTwister
  "Experiment name - used for results subdirectory"
  name::String 
  "Process labels"
  proclabels::Vector{Symbol}
  "Average sampling time"
  Tav::Float64
  "Minimal time difference between samples"
  Tmin::Float64
  "Number of observations"
  Ns::Vector{Int}
  "Number of simulation runs"
  Nrun::Int
end

function Base.show(io::IO,x::BatchDataSettings)
  for k in fieldnames(BatchDataSettings)
    v = getfield(x,k)
    vp = k==:rng ? "$(typeof(v))(...)" : v
    print(io,@sprintf("% 10s : %s\n",k,vp))
  end
end

settings_resultsdir(name) = resultsdir("ICMLA",name)

"""
$SIGNATURES

Batch data generation
"""
function batch_data(S::BatchDataSettings)
  config = (proclabel=S.proclabels,N=S.Ns,i=1:S.Nrun)
  partial_spec = vec([(;(keys(config).=>vs)...) for vs in Iterators.product(config...)])

  # generate datas
  dfrows = Vector{Any}(undef,length(partial_spec))
  prog = Progress(10)
  @threads for index in 1:length(partial_spec)
    ps = partial_spec[index]
    spec = (rng=copy(S.rng),N=ps.N,proclabel=ps.proclabel,Tav=S.Tav,Tmin=S.Tmin)
    proc = testprocess(spec.proclabel)
    data = rand(S.rng,proc,spec.N,spec.Tav,spec.Tmin)
    dfrows[index] = merge((index=index,),spec,(data=data,))
    next!(prog)
  end
  df = DataFrame(dfrows)
  
  # Save to file
  f = joinpath(settings_resultsdir(S.name),"data.jld")
  R = (df=df,settings=S,status=systemstatus())
  serialize(f,R); printstyled("\nSaved to file:\n$f\n",color=:blue,bold=true)

  return R
end

"
Settings for batch maximum likelihood estimation

$FIELDS
"
Base.@kwdef struct BatchMlEstimation
  "Model order"
  p::Int
  "Time scale of interest (Kullback-Leiber Discrepancy)"
  Tkld::Float64
  "Kalman engines"
  engines::Vector
  "Initiate at actual process?"
  init_actuals::Vector{Bool}
end

function Base.show(io::IO,x::BatchMlEstimation)
  for k in fieldnames(BatchMlEstimation)
    v = getfield(x,k)
    print(io,@sprintf("% 20s : %s\n",k,v))
  end
end

"
$SIGNATURES

Single maximum likelihood estimate for batch estimation. Returns
`nothing` if estimation trace cannot be computed.
"
function batch_mle_trace_single(S::BatchMlEstimation,spec,row_data;throw_errors=false)
  # Initial estimators
  tor = MlEstimator()
  Ts1 = [0.5,1,2]*S.Tkld
  Ts2 = [0.5,1,2]*row_data.Tav
  Ts = sort(unique([Ts1;Ts2]))
  init_tors = default_init_tors(row_data.rng,Ts)

  if spec.init_actual
    proc = testprocess(row_data.proclabel)
    if !(proc isa SdeProcess)
      # cannot initiate from a non-SDE process such as the white noise or squared exponential
      return nothing
    end
    push!(init_tors,InitialModel(proc))
  end

  problem = ProblemLike(row_data.data;p=S.p,engine=spec.engine)

  _,trace = estimate(tor,problem;init_tors=init_tors,ret_trace=true,throw_errors=throw_errors)
  
  return trace
end

function batch_mle_trace_body(S::BatchMlEstimation,df_data,specs)
  Ndata = nrow(df_data)
  df_arr = Array{Any,2}(undef,Ndata,length(specs))
  df_arr[:] .= missing
  @threads for idata in 1:Ndata
    row_data = df_data[idata,:]
    @threads for i in 1:length(specs)
      spec = specs[i]
      mle_trace = batch_mle_trace_single(S,spec,row_data)
      if !isnothing(mle_trace)
        df_arr[idata,i] = merge((index=row_data.index,),spec,(mle_trace=mle_trace,))
      end
    end
    print("[$(threadid())] $idata/$Ndata, ")
  end
  df = DataFrame(skipmissing(vec(df_arr)))
  return df
end

"
$SIGNATURES

Batch Maximum likelihood trace
"
function batch_mle_trace(S::BatchMlEstimation,df_data,results_dir)
  config = (init_actual=S.init_actuals,engine=S.engines)
  specs = vec([(;(keys(config).=>vs)...) for vs in Iterators.product(config...)])
  printstyled("$(length(specs)) Maximum likelihood configurations:\n",color=:blue)
  println.(specs)
  Ndata = nrow(df_data); Nspec = length(specs)
  t = Ndata*Nspec
  println("ML configurations: $Nspec Dataset count : $Ndata ⟹  Total estimates : $t")

  df = batch_mle_trace_body(S,df_data,specs)
  
  # Save to file
  f = joinpath(results_dir,"mle-trace.jld")
  output_mle_trace = (df=df,settings=S,status=systemstatus())
  serialize(f,output_mle_trace)
  printstyled("\nSaved to file:\n$f\n",color=:blue,bold=true)

  return output_mle_trace 
end

function batchsettings(f)
  bs = OrderedDict{Symbol,Any}()
  for (k,v) in TOML.parsefile(f)
    k = Symbol(k)
    if k==:proclabel v=Symbol(v)  end
    bs[k] = v
  end
  batchset = (;bs...)
  return batchset
end

function batch_posterior_samples(S,output_mle)
  estimates = output_mle.estimates
  # Tav = output_mle.settings.settings_data.Tav
  ## Body
  Nest = length(estimates)
  if !isnothing(S.Nest)
    @info "Sampling is done for $(S.Nest) out of $Nest maximum likelihood estimates"
    Nest = S.Nest
  end
  samples = Vector{Any}(undef,Nest)
  @threads for iest in 1:Nest
    estml = estimates[iest]
    data_c = estml.data_c
    tor = PmeanEstimator(;rng=data_c.rng,S.tor_kwargs...)
    proc_ml = estml.rmle.est
    samples[iest] = if proc_ml isa WhiteNoise
      (estml=estml,procs=nothing,samtrace=nothing)
    else
      p = EstimateSDE.order(proc_ml)
      problem = ProblemProb(data_c.data;p=p,engine=SdeRootsEngine,prior=ReferencePrior(Tav))
      _,procs,samtrace = estimate(tor,problem,proc_ml)``
      (estml=estml,procs=procs,samtrace=samtrace)
    end
    @info "Sampling for $iest completed"
  end

  # Save to file
  # TODO Make resdir an input argument to the function
  resdir = settings_resultsdir(output_mle.settings.settings_data)
  f = joinpath(resdir,"posterior-samples.jld")
  R = (samples=samples,settings=S,status=systemstatus())
  serialize(f,R)
  printstyled("\nSaved to file:\n$f\n",color=:blue,bold=true)

  return R
end

colors = [:blue, :light_red, :cyan, :magenta, :green, :light_blue, :light_magenta,
    :light_cyan,:light_green, :light_yellow,  :red, :yellow]
function tree(t::NamedTuple,depth=1;ind="————",maxdepth=Inf)
  color = colors[depth]
  identation = repeat(ind,depth)*" "
  for (k,v) in pairs(t)
    printstyled("$identation$k = ";color=color)
    if v isa NamedTuple
      if depth==maxdepth printstyled("…\n";color=color); return end
      printstyled("(\n";color=color) 
      tree(v,depth+1;ind=ind,maxdepth=maxdepth)
      printstyled("$identation)\n";color=color) 
    else
      println(repr(v)[1:min(end,80)])
    end
  end
  return nothing
end

"
$SIGNATURES

Estimates derived from maximum likelihood traces using `trace_estimates`.
"
function derived_estimates(df_mle_trace,results_dir)
  deriv_row = []
  for row_mle_trace in eachrow(df_mle_trace)
    table_trace_estimate = trace_estimates(row_mle_trace.mle_trace)
    for r in table_trace_estimate
      ei = r.estinfo
      tinfo = (p=ei.p,init_tor=ei.init_tor,est=ei.est)
      row_deriv = merge((;row_mle_trace...),(method=r.method,pmethod=r.pmethod),tinfo)
      push!(deriv_row,row_deriv)
    end
  end
  df = DataFrame(deriv_row)

  ## Save to file
  f = joinpath(results_dir,"derived-estimates.jld")
  R = (settings=nothing,df=df,status=systemstatus())
  serialize(f,R)
  printstyled("\nSaved to file:\n$f\n",color=:blue,bold=true)

  return R
end

"
$SIGNATURES

Batch model evaluation
"
function batch_model_evaluation(df_data,Smle_trace,df_derived,results_dir)
  ## KLD per estimate
  Nderived = nrow(df_derived)
  Nmetrics = 3 # Number of metrics
  rows_eval = Array{Any,2}(undef,Nmetrics,Nderived)
  @threads for i_derived in 1:Nderived
    row_derived = df_derived[i_derived,:]
    row_data = df_data[row_derived.index,:]
    est = row_derived.est
    proc = testprocess(row_data.proclabel)

    for i_metric in 1:Nmetrics
      label,T,t_kld = if i_metric==3
        t = getfield.(row_data.data,:t)
        "same",missing,t
      else
        T = i_metric==1 ? Smle_trace.Tkld : row_data.Tav
        duration = row_data.Tav*row_data.N
        t = (0:T:duration)
        "regular",T,t
      end
      kld = safe_kldiv(proc,est,t_kld,missing)
      row_eval = merge(row_derived,(label=label,T=T,kld=kld))
      rows_eval[i_metric,i_derived] = row_eval
    end
    print("[$(threadid())] $i_derived/$Nderived, ")
  end
  df = DataFrame(vec(rows_eval))

  ## Save to file
  f = joinpath(results_dir,"model-evaluation.jld")
  R = (settings=nothing,df=df,status=systemstatus())
  serialize(f,R); printstyled("\nSaved to file\n$f\n: ",color=:blue,bold=true)

  return R
end

const icmla_paper_dir = abspath(joinpath(@__DIR__,"..","ICMLA2020"))

"Copy to ICMLA paper figures directory"
function copy_icmla_figures(figure_dir,choice=nothing)
  if isnothing(choice)
    choice = request("Copy figures to ICMLA paper figures directory?",RadioMenu(["yes","no"]))
  end
  if choice==1
    dest_dir = joinpath(icmla_paper_dir,"figures")
    for fn in readdir(figure_dir)
      if !(endswith(fn,"pdf") || endswith(fn,"tex")) continue end
      src = joinpath(figure_dir,fn)
      if !isfile(src) continue end
      dst = joinpath(dest_dir,fn)
      cp(src,dst,force=true)
      println("Copied to ",dst)
    end
    println("\nDone.")
  end
  return choice
end
