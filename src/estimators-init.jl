## Initial parameter estimators ——————————————————————
"
$SIGNATURES

Convert autoregressive prediction parameters `a_ar` estimated from regularly sampled
data at interval `Tr` to SDE parameters `a_sde`. If `rm_negreal` is true,
remove negative real roots; otherwise, throw an error. 

Returns SDE parameters `a_sde`, and `negreal_count`, the number of negative real roots.
If `remove_negreal` is `false`, only `a_sde` is returned.
"
function ar2sdepar(a_ar,Tr,remove_negreal)
  p = [one(eltype(a_ar));-a_ar]
  pol = Polynomial(p[end:-1:1])
  rd = roots(pol)
  idx_negreal = map(x->isreal(x) && real(x)<0,rd)
  negreal_count = count(idx_negreal)
  if !remove_negreal && negreal_count>0 error("Negative real roots") end
  rd = rd[.!idx_negreal]
  a_sde = if isempty(rd)
    Float64[]
  else
    r = log.(rd)/Tr
    roots2sdepar(r,def_tol)
  end
  return remove_negreal ? (a_sde,negreal_count) : a_sde
end

is_expanding(problem,proc_prevs′) = problem.p>maximum(order.(proc_prevs′))

abstract type InitialEstimate end

"
Add random real root from an interval

$FIELDS
"
struct AddRoot<:InitialEstimate
  "root magnitude interval"
  i::Interval{Float64}
  "Random number generator"
  rng::MersenneTwister
  function AddRoot(i,rng)
    if !(i≫0) error("Root magnitude interval must be ≫ 0") end
    return new(i,rng)
  end
end

Base.show(io::IO,x::AddRoot) = print(io,"root ~ $(x.i)")

const default_addroot_int = 10.0..100.0

const default_addroot_rng = MersenneTwister(463468)

AddRoot() = AddRoot(default_addroot_int,copy(default_addroot_rng))

randroots(m::AddRoot,n::Int) = -rand(m.rng,Uniform(m.i.first,m.i.last),n)

function initial_estimate(m::AddRoot,problem::ProblemLike,proc_prevs)
  if !is_expanding(problem,proc_prevs) return nothing end
  proc = last(proc_prevs)
  n = problem.p-order(proc)
  r = [SdeRoots(proc).r;
       randroots(m,n)]
  eq = SdeRoots(sortroots(r);σy=problem.σy)
  return SdeProcess(eq)
end

"Mode for adding complex roots"
@enum ComplexRootMode KEEP_PREVIOUS MAX_COMPLEX ALL_RANDOM

"
Add random complex root from an interval

$FIELDS
"
struct AddComplexRoot<:InitialEstimate
  "Real part"
  i_re::Interval{Float64}
  "Imaginary part"
  i_im::Interval{Float64}
  "Root mode"
  rootmode::ComplexRootMode
  "Random Number generator"
  rng::MersenneTwister
  function AddComplexRoot(i_re,i_im,rootmode,rng)
    if !(i_im≫0) error("Imaginary part interval must be ≫ 0") end
    return new(i_re,i_im,rootmode,rng)
  end
end

function Base.show(io::IO,x::AddComplexRoot)
  print(io,x.rootmode," root ~ $(x.i_re)+$(x.i_im)im")
end

const default_addimagroot_int = 1e-2..10

function AddComplexRoot(i_re=default_addroot_int,i_im=default_addimagroot_int)
  rng = copy(default_addroot_rng)
  return AddComplexRoot(i_re,i_im,MAX_COMPLEX,rng)
end

function randroots(m::AddComplexRoot,n::Integer)
  nc = n÷2
  rc = Vector{ComplexF64}(undef,2nc)
  for i in 1:nc
    a = -rand(m.rng,Uniform(m.i_re.first,m.i_re.last))
    b = rand(m.rng,Uniform(m.i_im.first,m.i_im.last))
    rc[2(i-1)+1] = a-b*im
    rc[2(i-1)+2] = a+b*im
  end
  nr = n-2nc
  rr = -rand(m.rng,Uniform(m.i_re.first,m.i_re.last),nr)
  return [rc;rr]
end

function initial_estimate(m::AddComplexRoot,problem::ProblemLike,proc_prevs)
  if !is_expanding(problem,proc_prevs) return nothing end
  eq = if m.rootmode==KEEP_PREVIOUS
    # Add a complex root(s) to a previous model, keeping both complex and real roots
    proc = nothing
    for pp in reverse(proc_prevs)
      p = order(pp)
      if iseven(problem.p-p)
        proc = pp; break
      end
    end
    if isnothing(proc) return nothing end
    n = problem.p-order(proc)
    r = [SdeRoots(proc).r;randroots(m,n)]
    σy = std(getfield.(problem.data,:y);mean=0) # std(proc)
    SdeRoots(sortroots(r);σy=σy)
  elseif m.rootmode==MAX_COMPLEX
    # Add to complex roots of previous model, thereby maximizing the number of complex roots
    proc = proc_prevs[end]
    r = copy(SdeRoots(proc).r)
    r = filter(x->!isreal(x),r)
    n_add = problem.p-length(r)
    append!(r,EstimateSDE.randroots(m,n_add))
    SdeRoots(sortroots(r);σy=problem.σy)
  elseif m.rootmode==ALL_RANDOM
    # Purely random complex roots
    r = randroots(m,problem.p)
    y = getfield.(problem.data,:y)
    SdeRoots(sortroots(r))
  else error("Unknown root mode $(m.rootmode)")
  end
  return SdeProcess(sde(problem,eq))
end

"
Initial model returned only when a model of `order(proc)` is requested,
`nothing` otherwise.

$FIELDS
"
struct InitialModel{T<:SdeProcess} <:InitialEstimate
  "process"
  proc::T
end

function initial_estimate(m::InitialModel,problem::Problem,proc_prevs′)
  proc = if is_expanding(problem,proc_prevs′) && order(m.proc)==problem.p
    m.proc else nothing end
  return proc
end

"
Autoregressive models from resampled signals. Return the model with the largest
log likelihood. Does not use `proc_prev`.
Negative real roots are replaced by SDE roots using `Addroot`.

$FIELDS
"
struct ResampleAr<:InitialEstimate
  "Model sampling intervals"
  Ts::Vector{Float64}
  "Interpolation method"
  imethod::Symbol
  "Additional roots"
  addroot::AddComplexRoot
end

function initial_estimate(m::ResampleAr,problem::ProblemLike,proc_prevs::Any)
  if !is_expanding(problem,proc_prevs) return nothing end
  p = problem.p
  t = [x.t for x in problem.data]; y = [x.y for x in problem.data]
  Tmin = minimum(m.Ts)
  _,yr0 = interpolate(t,y,Tmin;method=m.imethod)
  proc_prev = proc_prevs[end]

  # Autoregressive models from resampled data
  procs = map(m.Ts) do T
    yr = T==Tmin ? yr0 : resample(yr0,Tmin/T)
    arhat,P,_ = estimate(ArModelStruct(p),reshape(yr,length(yr),1))
    a = predpar(arhat)[1][1,1,:,1]
    a_sde,n = ar2sdepar(a,T,true)
    r = isempty(a_sde) ? ComplexF64[] : SdeRoots(Sde(a_sde)).r
    rn = randroots(m.addroot,n)
    eq = SdeRoots(sortroots([r;rn]),problem.σy)
    SdeProcess(eq)
  end

  # Select model with maximum likelihood
  L = map(x->logpdf(problem,x),procs)
  eq = procs[argmax(L)].eq

  return SdeProcess(sde(problem,eq))
end

struct RemoveRoots<:InitialEstimate
  addroot::AddRoot
end

function initial_estimate(m::RemoveRoots,problem::ProblemLike,proc_prevs)
  if is_expanding(problem,proc_prevs) return nothing end
  proc_prevs = filter(x->order(x)>problem.p,proc_prevs)
  Lmax = -Inf; eq = nothing;
  for proc in proc_prevs
    r = roots(proc.eq)
    rs = r[end-problem.p+1:end]
    if imag(rs[1])>0  rs[1] = randroots(m.addroot,1)[1] end
    eq_c = SdeRoots(sortroots(rs);σy=problem.σy)
    L = logpdf(problem,SdeProcess(eq_c))
    if L>Lmax  eq = eq_c; Lmax = L  end
  end
  if isnothing(eq) return nothing end
  proc = SdeProcess(sde(problem,eq))
  return proc
end

function default_init_tors(rng::MersenneTwister,Ts::Vector{<:Number},n_random_init=5)
  addcomplex = AddComplexRoot(default_addroot_int,default_addimagroot_int,MAX_COMPLEX,rng)
  addroot = AddRoot(default_addroot_int,rng)
  init_tors = [
    addroot,
    addcomplex,
    ResampleAr(Ts,:linear,addcomplex),
    ResampleAr(Ts,:nearest,addcomplex),
    RemoveRoots(addroot)
  ]
  append!(init_tors,default_random_init_tors(rng,n_random_init))
  return init_tors
end

"
$SIGNATURES

Default initial estimators for `problem`. `T` is the time scale of interest. If omitted,
it is derived from the problem data using `minimal_time`.
"
function default_init_tors(problem::Problem,T::Number=minimal_time(problem))
  Ts = [0.5,1,2]*T
  @warn "Using system randomness for random number generator"
  return default_init_tors(MersenneTwister(),Ts)
end

"
$SIGNATURES

Default random initial estimators
"
function default_random_init_tors(rng::AbstractRNG,N::Int)
  init_tors = map(1:N) do _
    rng_c = MersenneTwister(rand(rng,UInt32,4))
    AddComplexRoot(default_addroot_int,default_addimagroot_int,ALL_RANDOM,rng_c)
  end
  return init_tors
end