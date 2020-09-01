"
Interface to PyStan to run Stan scripts
"
module PyStan

using MD5
using PyCall

export StanModel_cache, extract_samples

const pickle = PyNULL()
const pystan = PyNULL()
function __init__()
  copy!(pickle,pyimport_conda("pickle","pickle"))
  copy!(pystan,pyimport_conda("pystan","pystan"))
end

basedir = abspath(joinpath(@__DIR__,".."))

standir = joinpath(basedir,"stan")

"""StanModel_cache(model_name,model_code=nothing,cache_dir=joinpath(standir,"cache"),kwargs...) -> StanModel

Use just as you would `StanModel`. `kwargs` are optional keyword arguments for `StanModel`.

Returns a Python StanModel object.
"""
function StanModel_cache(model_name,model_code=nothing,cache_dir=joinpath(standir,"cache"),kwargs...)
  if isnothing(model_code)
    stanfile = joinpath(standir,"$model_name.stan")
    model_code = open(stanfile) do file
      read(file, String)
    end
  end
  code_hash = bytes2hex(md5(model_code))
  cache_fn = "cached-$model_name-$code_hash.pkl"
  if !isnothing(cache_dir)
    if !isdir(cache_dir) mkdir(cache_dir) end
    cache_fn = joinpath(cache_dir,cache_fn)
  end
  if isfile(cache_fn)
    sm = pickle.load(pybuiltin("open")(cache_fn,"rb"))
  else
    sm = pystan.StanModel(model_code=model_code,model_name=model_name,kwargs...)
    @pywith pybuiltin("open")(cache_fn,"wb") as f begin
      pickle.dump(sm,f)
  end
  end
  sm
end

# Although the documentation claims it is possible, I have had
# no luck in extracting a Dict in PyStan when permuted=false, hence this function
"""extract_dict(f,inc_warmup=false) -> Dict

Extract samples from a PyStan fit object. Also returns a dict if permuted is false
"""
function extract_samples(f;kwargs...)
  s = f.extract(;kwargs...)
  if !isa(s,Dict)
    n = [f.flatnames;"lp__"]
    s = Dict(k=>s[:,:,i] for (i,k) in enumerate(n)) 
  end
  s
end
end # module

using .PyStan

"""
$SIGNATURES

Input data for the stan script "sde.stan"
"""
function standata(problem::ProblemLike{S,M}) where {S,M}
  @unpack t,y,p = problem
  eststate = S isa Val{:est}
  S isa Val{:fixed} && @warn "SDE estimation with :fixed state not yet implemented"
  method = if M isa Val{:elimQ}
    error("Not implemented yet")
  elseif M isa Val{:matfrac}
    1
  end
  D = Dict(
    "N"=>length(t),
    "p"=>p,
    "t"=>t,
    "y"=>y,
    "eststate"=>Int64(eststate),
    "ref_prior_sigma_eps"=>Int64(false),
    "Tp"=>0,
    "method"=>method
  )
  if p>1
    D["p"] = p
  end
  return D
end

function standata(problem::ProblemProb)
  @unpack ref_prior_σϵ,Tp = problem
  D = standata(likeproblem(problem))
  D["ref_prior_sigma_eps"] = Int64(ref_prior_σϵ)
  D["Tp"] = isnothing(Tp) ? 0 : Tp
  return D
end

"""
$SIGNATURES
Stan parameters for process `proc`. `eststate` indicates whether
the state will be estimated.
"""
function stanpar(proc,emode,method)
  eq = proc.eq;
  spar =  Dict(
    "a" => eq.a,
    "sigma_eps" => eq.σϵ,
    "estinit" => Int64(emode isa Val{:est})
  )
  if emode isa Val{:est}
    proc.state isa State || error("State must be fixed Only fixed state (zero P) or no state is supported")
    spar["z2"] = proc.state.z[2:end]
  end
  return spar
end

function _modelsamples_stan_permuted(samples)
  S = length(samples["sigma_eps"])
  p = size(samples["a"],2)
  estinit = haskey(samples,"z")
  procsam = map(1:S) do i
    s = Dict(k=>v[i,:] for (k,v) in samples)
    eq = Sde(s["a"],s["sigma_eps"][1])
    state = estinit ? State(s["z"]) : nothing
    SdeProcess(eq,state)
  end
  return procsam
end

function _modelsamples_stan(samples)
  p = count(x->startswith(x,"a"),keys(samples))
  S = length(samples["sigma_eps"])
  procsam = map(1:S) do i
    s = Dict(k=>v[i] for (k,v) in samples)
    a = map(pc->s["a[$pc]"],1:p)
    eq = Sde(a,s["sigma_eps"])
    eststate = any(startswith.(keys(samples),"z")) # Is this correct?
    state = if eststate
      z = map(pc->s["a[$pc]"],1:p)
      State(z)
    else
      statstate(eq)
    end
    SdeProcess(eq,state)
  end
  return procsam
end

"""
$SIGNATURES

Converts `samples` returned by PyStan to an array of SdeProcess.
`permuted` is the parameter used with StanModel.extract (default `true`)
to extract the samples. 
"""
modelsamples_stan(samples,permuted=true) = if permuted
    _modelsamples_stan_permuted(samples)
  else
    _modelsamples_stan(samples)
end

function process_from_stan(problem::Problem{S,M},spar) where {S,M}
  a = spar["a"][:] # [:] to convert 0-dim array that is returned for SDE(1)
  σϵ = spar["sigma_eps"][1]
  eq = Sde(a,σϵ)
  state = if S isa Val{:stat}
    statstate(eq)
  else
    state = S isa Val{:fixed} ? problem.state : State(spar["z"][:])
  end
  proc = SdeProcess(eq,state)
  return proc
end

function ml_estimate_init_stan(tor,problem::Problem{S,M},procstart) where {S,M}
  @info "Stan ML estimate is incorrect because of determinant adjustment that should not be used for ML" maxlog=1
  sm = StanModel_cache("sde")
  data = standata(problem)
  sparstart = stanpar(procstart,S,M)
  optim = sm.optimizing(data,init=sparstart,iter=tor.iter,as_vector=false)
  
  spar = optim["par"]
  est = process_from_stan(problem,spar)
  return est,optim
end

function posterior_samples_stan(tor,problem::ProblemProb{S,M},procstarts) where {S,M}
  sm = StanModel_cache("sde")
  Tav = mean(diff(problem.t))
  N = length(problem.y)
  Tp = isnothing(problem.Tp) ? nothing : Tav
  data = standata(problem)
  cstate = compatiblestate(problem.p,problem.y)
  eststate = S isa Val{:est}
  parinit = stanpar.(procstarts,eststate)
  seed = rand(tor.rng,UInt16)
  tor.chains>1 && error("tor.chains > 1 with Stan results in errors; currently not supported")
  fit = sm.sampling(data,seed=seed,init=parinit,iter=tor.iter,chains=tor.chains)
  info = (fit=fit,)
  samples = extract_samples(fit)
  procsam = modelsamples_stan(samples)
  return procsam,info
end
