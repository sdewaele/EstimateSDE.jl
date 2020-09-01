# Process generation
# Power spectrum
# Autocovariance function

function Base.rand(rng::AbstractRNG,proc::TimeSeriesModel,N::Integer,Δt_av=1.0,Tmin=0.01)
  t = randtimes(rng,Δt_av,N,Tmin)
  return rand(rng,proc,t)
end

function Base.rand(rng::AbstractRNG,e::SdeEngine,state::State,t::AbstractVector)
  eq = e.eq
  N = length(t)
  p = statedim(eq)
  if measdim(eq)≠1 error("Only 1-D observations are supported") end
  data = Vector{Obs}(undef,N)

  z = rand(rng,state)
  μy,σ2y = obsdist(e,State(z))
  yc = σ2y==0 ? μy : rand(rng,Normal(μy,√σ2y))
  data[1] = Obs(t[1],yc)

  for i in 2:N
    Δt = t[i]-t[i-1]
    state = predict(e,State(z),Δt)
    z = rand(rng,state)
    μy,σ2y = obsdist(e,State(z))
    yc = σ2y==0 ? μy : rand(rng,Normal(μy,√σ2y))
    data[i] = Obs(t[i],yc)
  end
  return data
end

function Base.rand(rng::AbstractRNG,proc::TimeSeriesModel,t::AbstractVector)
  Σ = datacov(proc,t)
  L = cholesky(Σ).L
  ϵ = randn(rng,length(t)) # Not using MvNormal because it does not accept `Σ::SymmetricToeplitz`
  y = L*ϵ
  data = [Obs(tc,yc) for (tc,yc) in zip(t,y)]
  return data
end

"
$SIGNATURES

Default Kalman engine for process `proc`
"
defaultengine(proc::SdeProcess{<:Any,<:Any}) = ElimQ(proc.eq)

defaultengine(proc::SdeProcess{<:Any,<:SdeRoots}) = SdeRootsEngine(proc.eq)

"""
    rand(rng,proc,t) -> Vector(VectorObservation)

Random draw from the stationary vector SDE process `proc` at times `t`.
"""
function Base.rand(rng::AbstractRNG,proc::SdeProcess,t::AbstractVector)
  proc′ = transform_ss(Sde,proc)
  e = ElimQ(proc′.eq)
  return rand(rng,e,proc′.state,t)
end

"""
$SIGNATURES

Autocovariance function of state `z` at lags `τ`
"""
function autocovz(eq::VectorSde,τ)
  Rz0 = statcovz(eq)
  d = statedim(eq)
  Rz = similar(Rz0,d,d,length(τ))
  for (i,tc) in enumerate(τ)
    Rz[:,:,i] = Rz0*exp(eq.A*tc)'
  end
  Rz
end

"""
$SIGNATURES

Autocovariance function of observations `y` at lags `τ`
"""
function autocov(eq::VectorSde,τ)
  d = measdim(eq)
  Ry = similar(eq.R,d,d,length(τ))
  Rz = autocovz(eq,τ)
  for i in 1:length(τ)
    Ry[:,:,i] = eq.C*Rz[:,:,i]*eq.C'+eq.R
  end
  Ry
end

autocov(eq::Sde,τ) = autocov(convert(VectorSde,eq),τ)[:]

autocov(eq::SdeRoots,τ) = real.(autocov(convert(VectorSde,eq),τ)[:])

autocov(proc::SdeProcess,τ) = autocov(proc.eq,τ)

autocov(m::WhiteNoise,τ) = [τc==0 ? m.σ^2 : zero(m.σ) for τc in τ]

powerspectrum(m::WhiteNoise,f) = zero(f)

"""
$SIGNATURES

Power spectrum of Stochastic Differential Equation `eq`
at frequencies `f`
"""
function powerspectrum(eq::VectorSde,f)
  N = length(f)
  p = statedim(eq)
  q = measdim(eq)
  S = zeros(Complex{Float64},q,q,N)
  for (i,fc) in enumerate(f)
    ω = 2π*fc
    Sz = (eq.A-im*ω*I)\eq.Q/(eq.A-im*ω*I)'
    S[:,:,i] = eq.C*Sz*eq.C'
  end
  S
end

function powerspectrum(eq::ScalarSde,f)
  mv = convert(VectorSde,eq)
  S = powerspectrum(mv,f)
  real.(S[:])
end

powerspectrum(proc::ScalarSdeProcess,f) = powerspectrum(proc.eq,f)

function Distributions.logpdf(proc::TimeSeriesModel,data)
  t = getfield.(data,:t); y = getfield.(data,:y)
  Σ = datacov(proc,t)
  N = length(data)
  lp = -1/2*logdet(Σ) - 1/2*dot(y,Σ,y) - N/2*log(2π)
  return lp
end
