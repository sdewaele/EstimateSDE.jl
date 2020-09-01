# SDE estimators

## Supporting functions ————————————————————————
"""
    logpdf_reg_refprior(a,T=1.0) -> q

Unnormalized log reference prior for normalized SDE(1) parameter `α`,
`α = a*T` regularly sampled at interval `T`

Implements eq:sde1-log-reg-samp-refprior in lyx doc
"""
logpdf_reg_refprior(α) = -1/2*log(exp(abs(2α))-1)

"Approximate uninformative prior on SDE parameters a"
sdeparprior(a,Tp) = 
  sum(logpdf_reg_refprior(a/Tp^i) for (i,a) in enumerate(a))

uniform_rand(rng,x,w) = rand(rng,Uniform(x-w,x+w))

normparstd(x) = exp(abs(x))*√(1-exp(-2abs(x)))

"
$SIGNATURES

Model variation around process `proc` with variations in the 
transformed domain of size `f`.
"
function modelvariation(rng,problem::ProblemLike,proc,f)
  𝓣 = transformation(problem,proc)
  θ = paramstuple(problem,proc)
  η = inverse(𝓣,θ)
  ηv = η.+f*rand(rng,length(η))
  θv = 𝓣(ηv)
  procv = SdeProcess(problem,θv)
  return procv
end

modelvariation(rng,problem::ProblemProb,proc,f) = modelvariation(rng,problem.like,proc,f)

modelvariation(rng,problem,proc,f,n) = map(1:n) do _
  modelvariation(rng,problem,proc,f)
end

"""
$SIGNATURES
Posterior mean
"""
function posterior_mean(sam::AbstractArray{SdeProcess{T,U}}) where {T,U}
  a =  mean(x->x.eq.a,sam)
  σϵ = mean(x->x.eq.σϵ,sam)
  eq = Sde(a,σϵ)
  procm = if V<:State
    z = mean(x->x.state.z,sam)
    SdeProcess(eq,State(z))
  else
    SdeProcess(eq)
  end
  return procm
end

## Types ————————————————————————
"""
Maximum Likelihood estimator
$FIELDS
"""
Base.@kwdef struct MlEstimator
  "Number of iterations"
  N::Int64 = 1000
  "Root truncation threshold"
  R_truncate::Float64 = 1e5
end

"""
Posterior Mean estimator
$FIELDS
"""
Base.@kwdef struct PmeanEstimator
  "Sampling random number generator"
  rng::MersenneTwister
  "Number of iterations"
  N::Int64 = 1000
  "Indication for number of warmup iterations"
  Nwarmup::Int64 = 1000
  "Number of chains"
  nchains::Int64 = 4
  "Size of model variation in the transformed (η) domain"
  fvar::Float64 = 0.1
end

## Supporting functions WhiteNoise —————————————————————
"
$SIGNATURES

`logpdf` of white noise model with analytical maximum likelihood estimate for `σϵ`.

Returns log probability `lp` and the maximum likelihood estimate `σϵ`. 
"
function logpdf_ml_white_noise(data)
  y = getfield.(data,:y)
  σ2 = mean(y.^2)
  Nl = length(y)
  lp = -1/2*(Nl*log(σ2)+Nl+Nl*log(2π))
  σ = √σ2
  return lp,σ
end

"""
$SIGNATURES

Estimate white noise parameter (standard deviation) from `data`.
"""
function Distributions.estimate(::Type{WhiteNoise},data)
  _,σ = logpdf_ml_white_noise(data)
  return WhiteNoise(σ)
end

## DynamicHMC supporting functions Sde —————————————————————

"
$SIGNATURES

logpdf with analytical maximum likelihood estimate for `σϵ`.
`z` is the initial state. If `nothing`, use stationary state.

Returns log probability `lp` and the maximum likelihood estimate `σϵ`. 
"
function logpdf_ml_σϵ(e::SdeEngine{T},state,data) where {T}
  TR = real(T)
  E = zero(TR); S = zero(TR)

  # First observation 
  t = data[1].t; y = data[1].y
  μy,σ2y = obsdist(e,state)
  fixed_init_state = σ2y==0
  if fixed_init_state
    if !(isapprox(y,μy,rtol=1e-6)) error("Distribution inconsistent with observations: y[1] = $(y[1]) ≠ μy = $μy while σ2y==0") end
  else
    if !(σ2y>0) throw(DomainError(σ2y,"σ2y<0")) end
    S += log(σ2y)
    E += (y-μy)^2/σ2y
    state = measure(e,state,y,μy,σ2y)
  end
 
  N = length(data)
  for n in 2:N
    t = data[n].t; y = data[n].y
    t_prev = data[n-1].t
    state = predict(e,state,t-t_prev)

    μy,σ2y = obsdist(e,state)
    if !(σ2y>0) throw(DomainError(σ2y,"σ2y<0")) end
    S += log(σ2y)
    E += (y-μy)^2/σ2y
    state = measure(e,state,y,μy,σ2y)
  end

  Nl = fixed_init_state ? (N-1) : N
  σ2ϵ = E/Nl
  lp = -1/2*(Nl*log(σ2ϵ)+S+Nl+Nl*log(2π))
  σϵ = √σ2ϵ
  return lp,σϵ
end

## General parameter estimation ————————————————————————

"
$SIGNATURES

Estimate the SDE(1) model.
"
function estimate_sde1(problem::Problem,R=100)
  @unpack data,state_est = problem
  t = [x.t for x in data]; y = [x.y for x in data]
  σy = std(y,corrected=false,mean=0)
  Tav = mean(diff(t))
  _,yr = interpolate(t,y,Tav;method=:nearest)
  yb = yr[1:end-1]; σb = sqrt(mean(yb.^2))
  yf = yr[2:end]; σf = sqrt(mean(yf.^2))
  ah = mean(yb.*yf)/(σb*σf)
  a = ah≤0 ? -R : -log(ah)/Tav
  σϵ = √(2a)*σy
  est = SdeProcess(Sde([a],σϵ))
  return est
end

"""
$SIGNATURES
Maximum likelihood estimate given initial process estimate `procstart`,
defined in the state space corresponding the problem's `SdeEngine`.
"""
function ml_estimate_init(backend,tor,problem,procstart′)
  if procstart′∉problem error("Process\n$procstart′\nnot in range of problem:\n$problem") end 
  ml_est_init_backend = if backend==:dynamichmc
    ml_estimate_init_optim
  elseif backend==:stan
    ml_estimate_init_stan
  end
  return ml_est_init_backend(tor,problem,procstart′)
end

@enum EstimationStage EXPAND TRUNCATE

"
Estimation information used in estimation trace

$FIELDS
"
Base.@kwdef mutable struct EstimateInfo
    "model order"
    p::Int
    "Estimation stage"
    stage::EstimationStage
    "Initial estimator"
    init_tor::Union{Nothing,InitialEstimate} = nothing
    "Initial estimate"
    proc_ini::Union{Nothing,TimeSeriesModel} = nothing
    "Likelihood initial model"
    L_ini::Float64 = -Inf
    "Optimization converged?"
    converged::Bool = false
    "Estimate"
    est::Union{Nothing,TimeSeriesModel} = nothing
    "Likelihood estimate"
    L::Float64 = -Inf
    "Fatal error during estimation"
    error = nothing
end



function Base.show(io::IO, ::MIME{Symbol("text/plain")},x::EstimateInfo)
  println(io,"EstimateInfo")
  for k in fieldnames(EstimateInfo)
    print(io,@sprintf("% 12s : %s\n",k,getfield(x,k)))
  end
end

"
$SIGNATURES

Estimation trace entry for initial estimate `proc_ini`. Modifies `Tc`.
"
function estimate_trace_init!(Tc::EstimateInfo,tor,problem,proc_ini)
  proc_ini′ = SdeProcess(sde(problem,proc_ini.eq))
  Tc.L_ini = logpdf(problem,proc_ini′)
  Tc.est,opt = ml_estimate_init(:dynamichmc,tor,problem,proc_ini′)
  Tc.converged = Optim.converged(opt)
  Tc.L = logpdf(problem,Tc.est)
  return nothing
end

"
$SIGNATURES

SDE estimate and, optionally, the estimation trace. Note that the returned estimate may be
of lower order than the requested `problem.p`, because a lower order model may have
a higher likelihood.
"
function Distributions.estimate(tor::MlEstimator,problem::ProblemLike;
      init_tors=default_init_tors(problem),ret_trace=false,throw_errors=false)

  data = problem.data
  
  trace = EstimateInfo[]

  # White noise model
  L,σ = logpdf_ml_white_noise(data)
  est_wn = WhiteNoise(σ)

  push!(trace,EstimateInfo(;p=0,stage=EXPAND,est=est_wn,L=L,converged=true))

  ml_order = TimeSeriesModel[est_wn] # Maximum likelihood per order

  for stage in (EXPAND,TRUNCATE)
    pcs = stage==EXPAND ? (1:problem.p) : (problem.p-1:-1:2)
    for pc in pcs
      prob = lowerorder(problem,pc)
      n_tors = length(init_tors)
      TA = Vector{Any}(missing,n_tors)
      # Data copies so that threads do not have to access shared memory ==> higher speed
      thprob = [deepcopy(prob) for _ in 1:nthreads()]
      @threads for i in 1:n_tors
        prob_c = thprob[threadid()]
        init_tor_c = init_tors[i]
        Tc = EstimateInfo(;p=pc,init_tor=init_tor_c,stage=stage)
        try
          proc_ini0 = initial_estimate(init_tor_c,prob_c,ml_order)
          if isnothing(proc_ini0) continue end
          Tc.proc_ini = bringin(problem,proc_ini0)
          estimate_trace_init!(Tc,tor,prob_c,Tc.proc_ini)
        catch e
          if throw_errors||(e isa MethodError) rethrow() else @debug("SDE($pc) $(typeof(init_tor_c)) $e"); Tc.error = e end
        end
        TA[i] = Tc
      end
      T = collect(skipmissing(TA))
      if isempty(T) continue end
      append!(trace,T)

      # Maximum likelihood for current order and stage
      i = argmax(getfield.(T,:L))
      est_c = T[i].est
      if !isnothing(est_c)
        push!(ml_order,est_c)
      end
    end
  end 

  ## Overall maximum likelihood model. It can be of an order lower than `problem.p`
  i = argmax(getfield.(trace,:L))
  est = trace[i].est

  return ret_trace ? (est,trace) : est
end

function tracetable(data,trace,proc,t_kld)
  Href = -logpdf(proc,data)
  R = map(trace) do t
    D = if isnothing(t.est)
      missing
    else
      safe_kldiv(proc,t.est,t_kld)
    end
    u = (
      Hhat = -t.L-Href,
      D = D,
      init_tor_type = string(typeof(t.init_tor)),
      stage_str = string(t.stage)
    )
    tt = (;(x=>getfield(t,x) for x in fieldnames(typeof(t)))...)
    merge(tt,u)
  end |> DataFrame

  return R,Href
end
 
maxlike_est(trace) = trace[argmax(getfield.(trace,:L))]

"
$SIGNATURES

Selected `trace` estimates.
"
function trace_estimates(trace)
  R = []
  pmax = maximum(getfield.(trace,:p))

  # Overall MLE
  estinfo = maxlike_est(trace)
  push!(R,(method="MLE",pmethod=pmax,estinfo=estinfo))

  # MLE using only estimates from expansion phase
  estinfo = maxlike_est(filter(x->x.stage==EXPAND,trace))
  push!(R,(method="MLE EXPAND",pmethod=pmax,estinfo=estinfo))

  # MLE for order `pmax_c`
  for pmax_c in 0:pmax
    estinfo = maxlike_est(filter(x->x.p≤pmax_c,trace))
    push!(R,(method="MLE pmax",pmethod=pmax_c,estinfo=estinfo))
  end
  
  ## MLE from random initiation
  trace_random = filter(trace) do x
    x.init_tor isa AddComplexRoot && x.init_tor.rootmode==EstimateSDE.ALL_RANDOM
  end
  estinfo = maxlike_est(trace_random)
  push!(R,(method="MLE init ALL_RANDOM",pmethod=pmax,estinfo=estinfo))
  
  return R
end

"""
$SIGNATURES

Sample SDE posterior
"""
function posterior_samples(backend,tor,problem,procstarts)
  posterior_samples_backend = if backend==:dynamichmc
    posterior_samples_dynamichmc
  elseif backend==:stan
    posterior_samples_stan
  end
  return posterior_samples_backend(tor,problem,procstarts)
end

"
$SIGNATURES

Posterior mean estimate `procm` for SDE estimation `problem` based on HMC samples from starting
points concentrated around `procstart`. The posterior mean is taken over the
parameterization corresponding to the `problem`, so either SDE coefficients `a` or roots `r`.
"
function Distributions.estimate(tor::PmeanEstimator,problem::ProblemProb,procstart)
  procstart′ = SdeProcess(sde(problem.like,procstart.eq))
  procstarts = modelvariation(tor.rng,problem.like,procstart′,tor.fvar,tor.nchains)
  procs,R = posterior_samples(:dynamichmc,tor,problem,procstarts)
  procm = mean(procs,problem)
  return procm,procs,R
end

## DynamicHMC / Optim backend ————————————————————————

function ml_estimate_init_optim(tor,problem::ProblemLike,procstart)
  𝓣 = transformation(problem,procstart)
  fg! = function(F,G,η)
    f(η) = -(problem∘𝓣)(η)
    neg_lp,B = Zygote.pullback(f,η)
    if !isnothing(G)
      grad = isnothing(B) ? 0 : B(1)[1] 
      G[:] .= isnothing(grad) ? 0 : grad
    end
    if !isnothing(F) return neg_lp end
    return nothing
  end
  θstart = paramstuple(problem,procstart)
  ηstart = inverse(𝓣,θstart)
  opt = optimize(Optim.only_fg!(fg!),ηstart,
    LBFGS(),Optim.Options(f_tol=1e-2,iterations=tor.N,store_trace=true))
  θ = 𝓣(Optim.minimizer(opt))
  est = SdeProcess(problem,θ)
  return est,opt
end

function posterior_samples_dynamichmc(tor,problem::ProblemProb,procstarts)
  R = Vector{Any}(undef,tor.nchains)
  
  warmup_stages = default_warmup_stages(;
    local_optimization=nothing,
    middle_steps=Int(round(tor.Nwarmup/7)),
    doubling_stages=3
  )

  # Create data copies so threads do not need to access shared memory
  # this should accelerate processing
  threadpars = map(1:nthreads()) do _
    prob = deepcopy(problem)
    𝓣 = transformation(prob,first(procstarts))
    P = TransformedLogDensity(𝓣,prob)
    ∇P = ADgradient(:Zygote,P)
    (problem=prob,𝓣=𝓣,∇P=∇P)
  end

  @threads for i in 1:tor.nchains
    tpar = threadpars[threadid()]
    procs = procstarts[i]
    θs = paramstuple(tpar.problem,procs)
    ηs = inverse(tpar.𝓣,θs)
    R[i] = DynamicHMC.mcmc_keep_warmup(tor.rng,tpar.∇P,tor.N,
      initialization=(q=ηs,),
      warmup_stages=default_warmup_stages(;local_optimization=nothing,M=Symmetric),
      reporter=LogProgressReport(i,100,600) ) # NoProgressReport
  end
  chains = vcat([r.inference.chain for r in R]...)
  𝓣 = transformation(problem,first(procstarts))
  procs = map(η->SdeProcess(problem,𝓣(η)),chains)
  
  return procs,R
end
