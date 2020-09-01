# Kalman likelihood

"""
Computation engine for a stochastic differential equation
"""
abstract type SdeEngine{T} end

"""
Kalman computational engine based on eliminating the Q-matrix in the vector SDE(1)
representation using the standard state space.
"""
struct ElimQ{T,U<:AbstractMatrix{T},V<:AbstractMatrix{T}} <: SdeEngine{T}
  eq::Sde{T}
  A::U
  Σs::V
end

"""
    $SIGNATURES

Sde type used in an SdeEngine
"""
sdetype(::Type{ElimQ}) = Sde

function ElimQ(eq::Sde)
  veq = convert(VectorSde,eq)
  Σs = statcovz(veq)
  A = veq.A
  return ElimQ(eq,A,Σs)
end

"""
Kalman computational engine based on the matrix fraction computation.
"""
struct MatFrac{T,U<:AbstractMatrix{T},V<:AbstractMatrix{T}} <: SdeEngine{T}
  eq::Sde{T}
  A::U
  M::V
end

function MatFrac(eq::Sde)
  veq = convert(VectorSde,eq)
  A = veq.A
  Q = veq.Q
  p = order(eq)
  # Array mutation not supported by Zygote; using buffer.
  # Z = zeros(eltype(A),p,p)
  # M = [A  Q 
  #      Z -A']
  M_buf = Zygote.bufferfrom(zeros(eltype(A),2p,2p))
  M_buf[1:p,1:p]     = A; M_buf[1:p,p+1:2p]     =  Q;
  M_buf[p+1:2p,p+1:2p] = -A';
  M = copy(M_buf)
  return MatFrac(eq,A,M)
end

sdetype(::Type{MatFrac}) = Sde

"""
Roots-based Kalman computational engine. Performs Kalman computations
based on the roots representation.
"""
struct SdeRootsEngine{T,T1,T2,T3} <: SdeEngine{T}
  eq::SdeRoots{T}
  A::T1
  U::T2
  Σs::T3
end

function SdeRootsEngine(eq::SdeRoots)
  A = Diagonal(eq.r)
  U = eigenvecA(eq)
  Σs = statcovz(eq,U)
  return SdeRootsEngine(eq,A,U,Σs)
end

struct VectorSdeEngine{T,T1} <: SdeEngine{T}
  eq::VectorSde{T}
  Σs::T1
end

SdeRootsEngine(eq::Sde) = SdeRootsEngine(SdeRoots(eq))

"SDE type corresponding to the engine type"
sdetype(::Type{SdeRootsEngine}) = SdeRoots

"Vector SDE based Kalman computation engine."
function VectorSdeEngine(eq::Sde)
  veq = convert(VectorSde,eq)
  Σs = statcovz(veq)
  return VectorSdeEngine(veq,Σs)
end

sdetype(::Type{VectorSdeEngine}) = VectorSde

const ENGINES = [VectorSdeEngine,ElimQ,MatFrac,SdeRootsEngine]

"Default SDE engine"
default_engine = SdeRootsEngine

statcovz(e::Union{SdeRootsEngine,ElimQ,VectorSdeEngine}) = e.Σs

statcovz(e::MatFrac) = statcovz(e.eq)

statstate(e::SdeEngine) = State(statcovz(e))

"""
$SIGNATURES

Distribution mean `μy` and variance `σ2y` of the current observation
for a probabilistic `state`. 
"""
function obsdist(e::SdeRootsEngine,state::State)
  μ = real(sum(state.μ))
  σ2 = real(sum(state.Σ))
  return μ,σ2
end

function obsdist(e::Union{ElimQ,MatFrac},state::State)
  μ = state.μ[1]
  σ2 = state.Σ[1,1]
  return μ,σ2
end

function obsdist(e::VectorSdeEngine,state::State)
  C = e.eq.C
  μ = (C*state.μ)[1]
  Σ = C*state.Σ*C'+e.eq.R
  σ2 = Σ[1,1]
  return μ,σ2
end

"""
$SIGNATURES

Measurement step: condition state distribution on the current observation
`yc`.
"""
function measure(e::SdeRootsEngine,state,yc,μy,σ2y)
  w = sum(state.Σ;dims=2)
  μ = state.μ.+w[:].*(yc-μy)./σ2y
  Σ = state.Σ.-w*w'./σ2y
  return State(μ,Σ)
end

function measure(e::Union{ElimQ,MatFrac},state,yc,μy,σ2y)
  Pc = state.Σ[:,1]
  μ = state.μ.+Pc[:].*(yc-μy)/σ2y
  Σ = state.Σ.-Pc*Pc'./σ2y
  return State(μ,Σ)
end

function measure(e::VectorSdeEngine,state,yc,μy,σ2y)
  C = e.eq.C
  K = state.Σ*C'/σ2y # In Murphy, σ2y = S
  μ = state.μ.+K[:]*(yc-μy)
  Σ = (I-K*C)*state.Σ
  return State(μ,Σ)
end

function predict(A::AbstractMatrix,Σs::AbstractMatrix,state::State,Δt)
  F = exp(A*Δt)
  μ = F*state.μ
  Σ = Σs-F*(Σs-state.Σ)*F'
  return State(μ,Σ)
end

predict(e::Union{SdeRootsEngine,ElimQ},state,Δt) = predict(e.A,e.Σs,state,Δt)

predict(e::VectorSdeEngine,state,Δt) = predict(e.eq.A,e.Σs,state,Δt)

function predict(e::MatFrac,state::State,Δt)
  p = size(e.A,1)
  μp = exp(e.A*Δt)*state.μ
  C0 = state.Σ
  # CD0 = [C0;I] # Fails in Zygote ≤ v0.4.20
  CD0 = [C0;eye(eltype(C0),p)] # works with Zygote
  CD = exp(e.M*Δt)*CD0
  C,D = CD[1:p,:],CD[p+1:end,:]
  Pp = C/D
  return State(μp,Pp)
end

function logpdf_state(e::SdeEngine,state,data)
  μy,σ2y = obsdist(e,state)
  yc = data[1].y
  lp = if σ2y==0
    isapprox(yc,μy,rtol=1e-6) || error("Distribution inconsistent with observations: y[1] = $(y[1]) ≠ μy = $μy while σ2y==0")
    zero(μy)
  else
    state = measure(e,state,yc,μy,σ2y)
    logpdf(Normal(μy,√σ2y),yc)
  end
  for n in 2:length(data)
    tc = data[n].t; yc = data[n].y
    Δt = tc-data[n-1].t
    state = predict(e,state,Δt)
    μy,σ2y = obsdist(e,state)
    state = measure(e,state,yc,μy,σ2y)
    lp += logpdf(Normal(μy,√σ2y),yc)
  end
  return lp,state
end

"""
$SIGNATURES
 log probability `lp` an SDE process `proc` for `data`.
"""
function Distributions.logpdf(proc::ScalarSdeProcess{<:Any,<:Sde},data)
  lp,_ = logpdf_state(ElimQ(proc.eq),proc.state,data)
  return lp
end

function Distributions.logpdf(proc::ScalarSdeProcess{<:Any,<:SdeRoots},data)
  lp,_ = logpdf_state(SdeRootsEngine(proc.eq),proc.state,data)
  return lp
end
