# Estimation problem
# A "problem" is the combination of data and model parameters.
# This naming is used in DynamicHMC. In Stan, this is called the data block

"
$TYPEDEF

SDE estimation problem containing both the data and model parameters needed for parameter
estimation.
"
abstract type Problem{E<:SdeEngine} end

const widest_parint = 1e-3..1e3

const widest_freqint = 1e-3..1e2

const default_N_min = 100

function minval_with_data(x,N)
  xm = median(x)
  if length(x)>N
    xs = sort(x)
    xm = min(xm,xs[N])
  end
  return xm
end

"
$SIGNATURES

Parameter interval for given data. Not tested!
"
function parint(data::Vector{Obs})
  @warn "parint from data untested"
  t = diff([x.t for x in data])
  Tmin = minval_with_data(diff(t),default_N_min)
  Tmax = maximum(t)-minimum(t)
  idata = (1/Tmax)..(1/Tmin)
  i = idata ∩ widest_parint
  return i
end

"
Likelihood problem for Stochastic Differential Equation

$FIELDS
"
struct ProblemLike{E}  <: Problem{E}
  "vector of (time,observation) tuples"
  data::Vector{Obs}
  "Model order"
  p::Int
  "Parameter interval"
  parint::Interval{Float64}
  "Frequency range, used for imaginary part of roots (SdeRoots only)"
  freqint::Interval{Float64}
  "Only allow sorted roots (SdeRoots only)"
  sorted_roots::Bool
  "Standard deviation of `y`. Is pre-computed for use in the likelihood computation"
  σy::Float64
end

function ProblemLike(data;p=1,
    parint=widest_parint,freqint=widest_freqint,sorted_roots=true,engine=default_engine,
    σy = √mean(x->x.y.^2,data))
  return ProblemLike{engine}(data,p,parint,freqint,sorted_roots,σy)
end

function Base.show(io::IO,p::ProblemLike{E}) where {E}
  print(io,"SDE($(p.p))") 
  print(io," | a ∈ ",p.parint)
  if sdetype(E)===SdeRoots
    print(io," f ∈ ",p.freqint, " sorted roots: ", p.sorted_roots ? "yes" : "no")
  end
  print(io," | engine : $E")
  Tav = diff(map(x->x.t,p.data)) |> mean
  print(io," || N = ",length(p.data),@sprintf(" Tav = %0.2g",Tav))
end

function bringin(i::AbstractInterval,x,f)
  if !(i>0) error("Currently only works for intervals>0") end
  if x≪i  i.first/f  elseif x≫i  f*i.last else x end
end

"
$SIGNATURES

Is process `proc` in the problem range?
"
function Base.in(proc::SdeProcess{<:Any,<:Sde},problem::Problem)
  return order(proc)==problem.p && all(proc.eq.a.∈problem.parint)
end

function bringin(problem,eq::Sde)
  ain = bringin.(problem.parint,eq.a,0.9)
  return Sde(ain,eq.σϵ)
end

function Base.in(proc::SdeProcess{<:Any,<:SdeRoots},problem::Problem)
  rr,ri = splitroots(proc.eq.r)
  isin = order(proc)==problem.p &&
    all(-rr.∈problem.parint) && all(-real(ri).∈problem.parint) &&
    all(-imag(ri).<2π*problem.freqint.last)
  return isin
end

function bringin(problem,eq::SdeRoots)
  rr,ri = splitroots(eq.r)
  rrin = -bringin.(problem.parint,-rr,0.9)
  riin = map(ri) do ric
    a = -bringin(problem.parint,-real(ric),0.9)
    b = -bringin(problem.freqint,-imag(ric),0.9)
    a+b*im
  end
  rin = combineroots(rrin,riin)
  return SdeRoots(rin)
end

"
$SIGNATURES

Bring SDE process `proc` into the range of `problem` to guarantee
`(bringin(problem,proc) ∈ problem) == true`
"
function bringin(problem,proc::SdeProcess)
  eq = sde(problem,proc.eq)
  eqin = bringin(problem,eq)
  return SdeProcess(eqin)
end

"
$SIGNATURES

`SdeProblemLike` for lower model order `pc`.
"
function lowerorder(p::ProblemLike{E},pc) where {E}
  return ProblemLike{E}(p.data,pc,p.parint,p.freqint,p.sorted_roots,p.σy)
end

## Probabilistic probem ——————————————————————————————————

abstract type Prior end

"Flat prior in the SDE coefficients `a`"
struct FlatCoefPrior<:Prior end

Base.show(io::IO,::FlatCoefPrior) = print(io,"Flat a")

(p::FlatCoefPrior)(::NamedTuple{(:a,)}) = 0

"
Reference Prior derived from regularly sampled SDE(1) process at sampling interval `T`

$FIELDS
"
struct ReferencePrior<:Prior
  "Sampling time interval"
  T::Float64
end

Distributions.logpdf(p::ReferencePrior,θ::NamedTuple) = -1/2*sum(@. log(1-exp(2θ.r_re*p.T)))

# Note: A nested call is required for writing the custom adjoint below
(p::ReferencePrior)(θ::NamedTuple) = logpdf(p,θ)

# Ensure that the adjoint for prior(θ) is real-valued
Zygote.@adjoint (p::ReferencePrior)(θ::NamedTuple{(:r_re,:r_im)}) = begin
  lp,Bc = Zygote.pullback(logpdf,p,θ)
  if isinf(lp) return lp,Bc end
  B = function(l̄)
      p̄,θ̄c = Bc(l̄)
      # Because the prior does not depend on r_im, we know the derivative w.r.t. it is `nothing`
      # By hard coding this, type stability is maintained, hopefully resulting in faster execution.
      θ̄ = (r_re=real.(θ̄c.r_re),r_im=nothing)
      return p̄,θ̄
    end
  return lp,B
end

Zygote.@adjoint (p::ReferencePrior)(θ::NamedTuple{(:r_re,)}) = begin
  lp,Bc = Zygote.pullback(logpdf,p,θ)
  if isinf(lp) return lp,Bc end
  B = function(l̄)
      p̄,θ̄c = Bc(l̄)
      θ̄ = (r_re=real.(θ̄c.r_re),)
      return p̄,θ̄
    end
  return lp,B
end

"""
Probabilistic Stochastic Differential Equation problem.

$FIELDS
"""
struct ProblemProb{E,P<:Prior} <: Problem{E}
  "Likelihood"
  like::ProblemLike{E}
  "Prior"
  prior::P
end

ProblemProb(data;prior=FlatCoefPrior(),kwargs...) = ProblemProb(ProblemLike(data;kwargs...),prior)

Base.show(io::IO,p::ProblemProb) = print(io,p.like," || prior : ",p.prior)

engine(::ProblemLike{E},eq::AbstractSde) where {E} = E(eq)

function Distributions.logpdf(problem::ProblemLike,proc::SdeProcess)
  proc′ = SdeProcess(sde(problem,proc.eq))
  e = engine(problem,proc′.eq)
  lp,_ = logpdf_state(e,proc′.state,problem.data)
  return lp
end

"
$SIGNATURES

Log probability for named parameter tuple `θ` with maximum likelihood
estimate for `σϵ`.
"
function logpdf_ml_σϵ(problem::ProblemLike,θ)
  eq1 = sde(problem,θ,1)
  state1 = statstate(eq1)
  e = engine(problem,eq1)
  lp,σϵ = logpdf_ml_σϵ(e,state1,problem.data)
  return lp,σϵ
end

function (problem::ProblemLike{E})(θ) where{E}
  if sdetype(E)===SdeRoots && problem.sorted_roots
    if !issorted_rootstuple(θ)
      @debug "Rejecting point because roots are not sorted" θ
      return -Inf
    end
  end
  lp = try
    logpdf_ml_σϵ(problem,θ)[1]
  catch e
    @debug "Rejecting point because logpdf_ml_σϵ(problem,$θ) throws $e"
    -Inf
  end
  return lp
end

function safe_problem_adjoint(problem,θ)
  lp,B0 = try 
    Zygote.pullback((p,t)->logpdf_ml_σϵ(p,t)[1],problem,θ)
  catch e
    @debug "∂logpdf_ml_σϵ(problem,$θ) throws $e"
    return -Inf,(_) -> nothing
  end
end

# Ensure that the adjoint for problem(θ) catches errors
Zygote.@adjoint (problem::ProblemLike)(θ) = safe_problem_adjoint(problem,θ)

# Ensure that the adjoint for problem(θ) catches errors and is real-valued
Zygote.@adjoint (problem::ProblemLike{SdeRootsEngine})(θ) = begin
  lp,Bc = safe_problem_adjoint(problem,θ)
  if isinf(lp) return lp,Bc end
  B = function(l̄)
      p̄,θ̄c = Bc(l̄)
      θ̄ = map(x->real.(x),θ̄c)
      return p̄,θ̄
    end
  return lp,B
end

transform_ss(problem::Problem,proc) = SdeProcess(sde(problem,proc.eq))

(problem::ProblemProb)(θ) = (problem.like)(θ)+(problem.prior)(θ)

"
Time interval for which sufficient data is available to estimate the autocorrelation.
"
function minimal_time(problem::ProblemLike,N=default_N_min)
  Δt = diff([x.t for x in problem.data])
  Tmin = minval_with_data(Δt,N)
  return Tmin
end

minimal_time(problem::ProblemProb) = minimal_time(problem.like)

function Statistics.mean(procs::Vector{<:SdeProcess},::Problem)
  eq = mean(getproperty.(procs,:eq))
  return SdeProcess(eq)
end