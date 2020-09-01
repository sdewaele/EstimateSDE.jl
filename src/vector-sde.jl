# First-order vector Stochastic Differential Equations

# NOTE: for scalar SDEs, checkstable can also be used
"""isstableode(A) -> Bool

Determine whether the matrix A corresponds to a stable ordinary differential equation
"""
isstableode(A) = all(real.(eigen(A).values).<0)

"""isdiagonalizable(A) -> Bool

Determine whether the matrix A is diagonalizable
"""
isdiagonalizable(A) = rank(eigen(A).vectors)==size(A,1)

abstract type TimeSeriesModel{T<:Number} end

abstract type AbstractSde{T<:Number} end

"""
Vector valued, first order stochastic differential equation with measurement model
$FIELDS
"""
struct VectorSde{T<:Number,U<:AbstractMatrix{T}} <: AbstractSde{T}
  A::U
  C::Matrix{T}
  Q::Matrix{T}
  R::Matrix{T}
  function VectorSde(A::U,C,Q,R) where {T,U<:AbstractMatrix{T}}
    p = LinearAlgebra.checksquare(A)
    q = size(C,1)
    size(C,2)==p || error("C must be q×",p)
    LinearAlgebra.checksquare(Q)==p || error("Q must be ",p,"×",p)
    LinearAlgebra.checksquare(R)==q || error("R must be ",q,"×",q)
    new{T,U}(A,C,Q,R)
  end
end

function Base.convert(::Type{AbstractSde{T}},x::VectorSde) where {T}
  return VectorSde(convert.(AbstractMatrix{T},[x.A,x.C,x.Q,x.R])...)
end

"State dimension"
statedim(m::VectorSde) = size(m.A,1)

"measurement dimension"
measdim(m::VectorSde) = size(m.R,1)

"Model order"
order(eq::VectorSde) = 1

isstable(eq::VectorSde) = isstableode(eq.A)

"""
    statcovz(m::VectorSde) -> R0

Stationary covariance matrix of the state variable `z`
"""
statcovz(m::VectorSde) = project_symmetric(lyap(m.A,m.Q))

"""
$SIGNATURES

Stationary covariance matrix of the observations `y`
"""
statcov(m::VectorSde) = m.C*statcovz(m)*m.C'+m.R

"""
Stationary state distribution
$SIGNATURES
"""
statstate(eq::AbstractSde) = State(statcovz(eq))

"""
    diag_vectorsde(m::VectorSde) -> VectorSde

Representation of the  vector SDE `m` with diagonalized `A` matrix.
Only if `scale` is `true, scale eigenvectors that the first component
is one.

NOTE: In general `diag_vectorsde` returns complex-valued models. Not all
functions work on complex-valued models, e.g. `rand`.
"""
function diag_vectorsde(m::VectorSde,scale=true)
  E = eigen(m.A)
  U = E.vectors
  if scale
    for j = 1:size(U,2)
      U[:,j] ./= U[1,j]
    end
  end
  A = Diagonal(E.values)
  C = m.C*U
  Q = U\m.Q/U'
  R = complex(m.R)
  VectorSde(A,C,Q,R)
end

function LinearAlgebra.lyap(A::Diagonal,C)
  a = A.diag
  p = length(a)
  ac = repeat(a,1,p)
  m = ac.+ac'
  -C./m
end

"Observation time and value"
struct Obs
  t::Float64
  y::Float64
end

## SDE process ————————————————————
"""
SDE process: Combines Stochastic Differential Equation with state
$FIELDS
"""
struct SdeProcess{T,U} <: TimeSeriesModel{T}
  "Stochastic Differential Equation"
  eq::U
  "State"
  state::State{T}
  function SdeProcess(eq::U,state::State{T}) where {T,U}
    if statedim(eq)≠dim(state) error("Model order mismatch between SDE and state") end
    new{T,U}(eq,state)
  end
end

SdeProcess(eq) = SdeProcess(eq,statstate(eq))

function Base.show(io::IO,proc::SdeProcess)
  print(io,proc.eq,"; state : ")
  if has_statstate(proc)
    print(io,"stationary")
  else
    print(io,proc.state)
  end
end

VectorSdeProcess = SdeProcess{T,U} where {T,U<:VectorSde}

function describe_model(proc::SdeProcess)
  describe_model(proc.eq)
  print("State: ")
  if has_statstate(proc)
    println("Stationary")
  else
    println(proc.state)
  end
end

order(proc::SdeProcess) = order(proc.eq)

Statistics.std(proc::SdeProcess) = std(proc.eq)

"Has stationary state?"
has_statstate(proc::SdeProcess;rtol=√eps()) = isstable(proc.eq) && isapprox(proc.state,statstate(proc.eq);rtol=rtol)

function Statistics.mean(procs::Vector{<:SdeProcess})
  eq = mean(getproperty.(procs,:eq))
  state = mean(getproperty.(procs,:state))
  return SdeProcess(eq,state)
end

