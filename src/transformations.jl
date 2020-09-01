# Parameter transformations

## TransformVariables transformations for θ <--> η
function transformation(problem::ProblemLike,eq::SdeRoots)
  rr,ri = splitroots(eq.r)
  ni = length(ri)
  nr = length(rr)+ni
  ir = problem.parint
  tr = as(Vector,as(Real,-ir.last,-ir.first),nr)
  return if ni==0
    as((r_re=tr,))
  else
    ii = problem.freqint
    as((r_re=tr,r_im=as(Vector,as(Real,-ii.last,-ii.first),ni)))
  end
end

function transformation(problem::ProblemLike,::Sde)
  i = problem.parint
  return as((a=as(Vector,as(Real,i.first,i.last),problem.p),))
end

transformation(problem::ProblemLike,proc::SdeProcess) = transformation(problem,proc.eq)

transformation(problem::ProblemProb,args...) = transformation(problem.like,args...)

## Equation to parameter tuples: proc <--> θ

"Estimation parameters `θ` for `TransformVariables`."
paramstuple(eq::Sde) = (a=eq.a,)

function paramstuple(eq::SdeRoots)
  rr,ri = splitroots(eq.r)
  r_re = [real(ri);real(rr)]
  θ = isempty(ri) ? (r_re=r_re,) : (r_re=r_re,r_im=imag(ri))
  return θ
end

paramstuple(::Problem,proc::SdeProcess) = paramstuple(proc.eq)

sde(::Problem{E},args...;kwargs...) where {E} = sdetype(E)(args...;kwargs...)

function SdeProcess(problem::ProblemLike,θ)
  _,σϵ = logpdf_ml_σϵ(problem,θ)
  proc = SdeProcess(sde(problem,θ,σϵ))
  return proc
end

SdeProcess(problem::ProblemProb,θ) = SdeProcess(problem.like,θ)

## Transform processes

# Some of the functionality `transform_ss` can probably more elegantly be handled by
# implementing `convert` and/or constructors combined with a function that transforms the state.

function transform_ss(D::Type,proc::SdeProcess{T,U}) where {T,U<:AbstractSde{T}}
  if !(U <: D)  error("transform_ss from $U to $D not defined")  end
  return proc
end

"
$SIGNATURES

Transform process `proc` to the state space of `D`.
"
function transform_ss(::Type{SdeRoots},proc::SdeProcess{<:Any,<:Sde})
  eq′ = SdeRoots(proc.eq)
  e = SdeRootsEngine(eq′)
  @unpack μ,Σ = proc.state
  μ′ = e.U\μ
  Σ′ = e.U\Σ/e.U'
  state′ = State(μ′,Σ′)
  proc′ = SdeProcess(eq′,state′)
  return proc′
end

function transform_ss(::Type{Sde},proc::SdeProcess{<:Any,<:SdeRoots},tol::Float64=def_tol)
  eq′ = Sde(proc.eq,tol)
  e = SdeRootsEngine(proc.eq)
  @unpack μ,Σ = proc.state
  μ′ = real(e.U*μ)
  Σ′ = real(e.U*Σ*e.U')
  state′ = State(μ′,Σ′)
  return SdeProcess(eq′,state′)
end

function transform_ss(::Type{VectorSde},proc::SdeProcess{<:Any,<:Sde})
  eq = convert(VectorSde,proc.eq)
  return SdeProcess(eq,proc.state)
end

function transform_ss(::Type{Sde},proc::SdeProcess{<:Any,<:VectorSde})
  a = -proc.eq.A[end,:]
  σϵ = √proc.eq.Q[end,end]
  eq′ = Sde(a,σϵ)
  return SdeProcess(eq′,proc.state)
end

# Custom adjoint for transform to guarantee real-valued adjoint
using TransformVariables:VectorTransform,TransformTuple,@argcheck,transform_with,NOLOGJAC

Zygote.@adjoint TransformVariables.transform(t::TransformTuple{<:Union{NamedTuple{(:r_re, :r_im)},NamedTuple{(:r_re,)}}},η) = begin
  θ,Bc = Zygote.pullback(t,η) do t,η
    # Code duplicate for `transform(t,η)`. We cannot use transform directly because the adjoint would call itself.
    # Is there a better way?
    @argcheck dimension(t) == length(η)
    first(transform_with(NOLOGJAC, t, η, firstindex(η)))
  end
  B = function(θ̄)
    t̄,η̄c = Bc(θ̄)
    η̄ = real.(η̄c)
    return t̄,η̄ 
  end
  return θ,B
end
