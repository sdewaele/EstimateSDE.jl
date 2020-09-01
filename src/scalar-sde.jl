# Higher-order scalar stochastic diffential equations

# Scalar SDE type ————————————————————————

abstract type ScalarSde{T} <: AbstractSde{T} end

"""Scalar stochastic differential equation (SDE)

The differential equation is given by:
    a[1]\\*y(t) + a[2]⋅(dy/dt)(t) + ⋯  + a[p]\\*(dᵖ⁻¹y/dᵖ⁻¹t) + (dᵖy/dᵖt)(t) = ϵ(t)
"""
struct Sde{T} <: ScalarSde{T}
  "Parameter vector"
  a::Vector{T}
  "Driving noise standard deviation"
  σϵ::T
  function Sde(a::Vector{T},σϵ::T) where {T}
    (isreal(σϵ) && real(σϵ)>0) || error("σϵ must be positive")
    new{T}(a,σϵ)
  end
end

function Sde(a::Vector{T},σϵ::U) where {T,U}
  V = promote_type(T,U)
  return Sde(convert(Vector{V},a),convert(V,σϵ))
end

Sde(θ::NamedTuple,σϵ) = Sde(θ.a,σϵ)

Sde(eq::Sde) = eq

function Sde(a;σy=one(eltype(a)))
  g = std(Sde(a,one(eltype(a))))
  σϵ = σy/g
  return Sde(a,σϵ)
end

function Base.convert(::Type{AbstractSde{T}},x::Sde) where {T}
  return Sde(convert(Vector{T},x.a),convert(T,x.σϵ))
end

function Base.show(io::IO,eq::Sde)
  p = order(eq)
  print(io,"SDE($p) ")
  cc = map(enumerate(eq.a)) do (i,a)
    ystr = i==1 ? "y" : @sprintf "dₜ%sy" expchar(i-1)
    @sprintf "%0.3g⋅%s" a ystr
  end
  cf = @sprintf "dₜ%sy" expchar(p)
  c = [cc;cf]
  lhs = join(c," + ")
  σϵstr = @sprintf "σϵ = %0.3g" eq.σϵ
  print(io,"$lhs = ϵ | $σϵstr")
end

function Base.show(io::IO,::MIME"text/plain",eq::Sde)
  print(io,"Scalar Stochastic Differential Equation\n",eq)
end

label(m::T) where T<:TimeSeriesModel = T.name

label(eq::ScalarSde) = "SDE($(order(eq)))"

order(eq::Sde) = length(eq.a)

statedim(eq::ScalarSde) = order(eq)

measdim(eq::ScalarSde) = 1

# Zygote-compatible code for a unit matrix
eye(T,n) = Matrix{T}(I,n,n)
Zygote.@nograd eye

function Base.convert(::Type{VectorSde},eq::Sde{T}) where {T}
  p = order(eq)
  # A = vcat([zeros(T,p-1,1) I],-eq.a') # Fails in Zygote v0.4.20
  A = vcat(hcat(zeros(T,p-1,1),eye(T,p-1)),-eq.a') # works with Zygote
  C = [one(T) zeros(T,1,p-1)]
  # Q = vcat(zeros(T,p-1,p),hcat(zeros(T,1,p-1),eq.σϵ^2)) # Fails in Zygote v0.4.20 
  Q = vcat(zeros(T,p-1,p),hcat(zeros(T,1,p-1),[eq.σϵ^2])) # works in Zygote
  R = zeros(T,1,1)
  return VectorSde(A,C,Q,R)
end

ScalarSdeProcess{T,U} = SdeProcess{T,U} where {T,U<:ScalarSde{T}}

function Base.convert(::Type{VectorSdeProcess},proc::ScalarSdeProcess)
  eq = convert(VectorSde,proc.eq)
  return SdeProcess(eq,proc.state)
end

label(proc::ScalarSdeProcess) = label(proc.eq)

# Method required to fix method ambiguity for `convert`.
Base.convert(::Type{T},proc::T) where T<:ScalarSdeProcess = proc

function Statistics.std(eq::Sde)
  mv = convert(VectorSde,eq)
  Σs = statcovz(mv)
  σy = √Σs[1,1]
  return σy
end

statcovz(eq::Sde) = statcovz(convert(VectorSde,eq))

function Statistics.mean(eqs::Vector{<:Sde})
  a = mean(getproperty.(eqs,:a))
  σϵ = mean(getproperty.(eqs,:σϵ))
  return Sde(a,σϵ)
end

"""
$SIGNATURES

Is `eq` a stable differential equation?
"""
isstable(eq::ScalarSde) = all(real.(roots(eq)).<0)

"""
$SIGNATURES

Check whether `eq` a stable differential equation. Raise error if not,
otherwise return `eq`.
"""
function checkstable(eq::Sde)
  isstable(eq) ||  error("Unstable model $eq")
  return eq
end

function expchar(i)
  expchars = collect("¹²³⁴⁵⁶⁷⁸⁹")
  i≤9 ? string(expchars[i]) : "($i)"
end

describe_model(m::TimeSeriesModel) = println(m)

"
$SIGNATURES

Describe a stochastic differential equation
"
function describe_model(eq::ScalarSde)
  print("$eq | ")
  p = order(eq)
  if isstable(eq)
    σy = std(eq)
    @printf "σy = %0.3g" σy
  else
    printstyled("unstable\n",color=:yellow)
  end
  r = roots(eq)
  println("\nRoots")
  printroots(r)
end
