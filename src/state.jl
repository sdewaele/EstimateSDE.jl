# SDE state

"""
Probabilistic SDE State
$FIELDS
"""
struct State{T<:Number}
  "mean z"
  μ::Vector{T}
  "covariance matrix"
  Σ::Matrix{T}
  function State(μ::Vector{T},Σ::Matrix{T}) where {T}
    p = length(μ)
    LinearAlgebra.checksquare(Σ)==p || error("Σ must be $p-by-$p")
    new{T}(μ,Σ)
  end
end

State(μ::Vector,Σ::Matrix) = State(promote(μ,Σ)...)

function Base.show(io::IO,ss::State{T}) where {T}
  elstr = if T<:Complex
    x->@sprintf("%0.3g+%0.3gim",real(x),imag(x))
  else
    x->@sprintf("%0.3g",x)
  end
  μs = elstr.(ss.μ)
  μstr = join(μs," ")
  Σstr = if iszero(ss.Σ)
    0
  else
    Σs = elstr.(ss.Σ)
    rows = join.(eachrow(Σs)," ")
    Σstr = join(rows,"; ")
    "["*Σstr*"]"
  end
  print(io,"μ = [",μstr,"] | Σ ",Σstr)
end

function Base.convert(::Type{State{T}},x::State) where {T}
  return State(convert(AbstractVector{T},x.μ),convert(AbstractMatrix{T},x.Σ))
end

Distributions.dim(state::State) = length(state.μ)

function Statistics.mean(states::Vector{<:State})
  μ = mean(getproperty.(states,:μ))
  Σ = mean(getproperty.(states,:Σ))
  return State(μ,Σ)
end

Base.rand(rng::AbstractRNG,state::State) = semiposdef_mvrand(rng,state.μ,state.Σ)

function State(Σ::AbstractMatrix{T}) where {T}
  p = size(Σ,1)
  μ = zeros(T,p)
  return State(μ,Σ)
end

function State(μ::AbstractVector{T}) where {T}
  p = length(μ)
  Σ = zeros(T,p,p)
  return State(μ,Σ)
end

import Base.:(==)
x::State==y::State = x.μ==y.μ && x.Σ==y.Σ

Base.hash(x::State, h::UInt) = hash(x.μ, hash(x.Σ, h))

Base.isapprox(x::State,y::State;kwargs...) = isapprox(x.μ,y.μ;kwargs...) && isapprox(x.Σ,y.Σ;kwargs...)

# """
# $SIGNATURES

# State of order `p`and compatible with data `y` and zero elsewhere
# """
# function compatiblestate(p,y)
#   z = similar(y,p)
#   z[:] .= 0
#   z[1] = y[1]
#   return State(z)
# end

# iscompatible(state::State,y) = !iszero(state.Σ) || state.μ[1]==y[1]

# function checkcompatible(state::State,y)
#   iscompatible(state,y) || error("state $state incompatible with data")
#   return state
# end

"
$(SIGNATURES)

Return deterministic state value. Throws an error if the state is not deterministic
"
function detstate(ss::State)
  if !iszero(ss.Σ)  error("Not a deterministic state")  end
  return ss.μ
end
