"System status"
function systemstatus()
  io = IOBuffer()
  versioninfo(io;verbose=true)
  vinfo = String(take!(io))
  close(io)

  status = (
    gitlog=read(`git log -1`,String),
    versioninfo=vinfo,
    datetime = now()
  )
  return status
end

project_symmetric(A) = (A+A')/2

"""
$SIGNATURES

Index in the at most `n` most frequent unique elements
of `x`. If the final number of unique elements is equal
to length(x), return `nothing`

Used in precomputation of matrix exponentials.
"""
function unique_index(x,n=length(x))
  n>0 || return nothing
  xu = unique(x)
  c = map(u->count(x.==u),xu)
  p = sortperm(c,rev=true)
  xus = xu[p]
  n = min(length(xus),n)
  xus = xus[1:n]
  length(xus)<length(x) || return nothing
  xus = xus[1:n]
  idx = indexin(x,xus)
  (idx=idx,x=xus)
end

"Is this code run from Visual Studio code?"
invscode() = occursin("code",ENV["JULIA_EDITOR"])

"""
    logpdf2pdf(lp) -> p

Convert logpdf to pdf.
"""
function logpdf2pdf(lp)
  u = exp.(lp.-maximum(lp))
  u./mean(u)
end

"
$SIGNATURES

Interpolate signal `t,x` at times `tr`.
"
function interpolate(t,y,tr;method=:nearest,extra=false)
  N = length(t)
  Nr = length(tr)
  yr = similar(y,Nr)
  issorted(t) || error("t not be sorted")
  i1 = 1; t1 = t[i1]; y1 = y[i1]
  i2 = 2; t2 = t[i2]; y2 = y[i2]
  for (i,τ) in enumerate(tr)
    if !(t[1]≤τ≤t[end])
      if !extra error("No extrapolation") end
      yr[i] = τ≤t[1] ? y[1] : y[end]
      continue
    end
    if τ>t2
      while t[i2]<τ i2+=1 end
      t2 = t[i2]; y2 = y[i2]
      i1 = i2-1
      while t[i1]>τ i1-=1 end
      t1 = t[i1]; y1 = y[i1]
    end
    yc = if method==:nearest
      t2-τ<τ-t1 ? y2 : y1
    elseif method==:linear
      (τ-t1)/(t2-t1)*(y2-y1)+y1
    else
      error("Unknown interpolation method $method")
    end
    yr[i] = yc
  end
  return yr
end

"
$SIGNATURES

Interpolate signal `t,x` at sampling interval `Tr`. Returns `tr,yr`.
"
function interpolate(t,y,Tr::Number;method=:nearest)
  tr = t[1]:Tr:t[end]
  yr = interpolate(t,y,tr;method=method)
  return tr,yr
end

"""
    randtimes([rng=GLOBAL_RNG,]Tav,n) -> t

Randomly distributed times
"""
function randtimes(rng::AbstractRNG,Tav,n,Tmin=0)
  Tavc = Tav-Tmin
  d = Exponential(Tavc)
  Δt = rand(rng,d,n).+Tmin
  t = cumsum(Δt)
end

randtimes(Tav,n) = randtimes(Random.GLOBAL_RNG,Tav,n)

"""
    semiposdef_mvrand(rng,μ,Σ)

Random draw from a multi-variate normal distribution with
a positive definite or positive semi-definite covariance matrix Σ
"""
function semiposdef_mvrand(rng,μ,Σ,tol=1e-10)
  r = if iszero(Σ)
    μ
  elseif isposdef(Σ)
    rand(rng,MvNormal(μ,Matrix(Σ)))
  else
    Qe = eigen(Σ)
    ϵ = map(Qe.values) do x
      if x≤0
          x >-tol || error("Σ has negative eigenvalue")
          0
      else
        √x*randn(rng)
      end
    end
    Qe.vectors*ϵ .+ μ
  end
  return r
end

"""
$SIGNATURES

Test process
"""
function testprocess(l)
  ## SDE roots based test processes
  r = if l==:A
    ComplexF64[0.2]
  elseif l==:B
    combineroots(Float64[],[-1/10 + 2π*0.28im])
  elseif l==:C
    combineroots([-1/18],[-1/30+2π*0.32im])
  elseif l==:D
    combineroots(Float64[],[-1/60 + 2π*0.11im,-1/22 + 2π*0.14im])
  elseif l==:E
    rr = [-1/18,-1/22,-1/140]
    ri = [-1/22 + 2π*0.14im,-1/60 + 2π*0.11im,-1/100 + 2π*0.04im]
    combineroots(rr,ri)
  elseif l==:ICMLA_A
    combineroots([-1/200],ComplexF64[])
  elseif l==:ICMLA_C
    combineroots(Float64[],[-0.10 + 2π*0.25im,-0.5 + 2π*1.5im])
  end

  if !isnothing(r)
    σy = l==:B ? 3.5 : 1
    proc = SdeProcess(SdeRoots(r;σy=σy))
    return proc
  end

  ## Other processes
  proc = if l==:ICMLA_B
    SquaredExp(0.3,1.0,0.01)
  elseif l==:ICMLA_W
    WhiteNoise()
  end

  if isnothing(proc) error("Unknown test process label $l") end

  return proc
end

"
Generate `N` observations of a discrete-time AR(1) process with parameter `a`

y[n] = a*y[n-1] + ϵ[n]

with unit standard deviation. This can be used in stress-testing SDE estimation
by generating data with negative parameter `a`, which results in an
oscillating signal that is not well described by an SDE(1) model. 
"
function ar1data(rng,a,N)
  σϵ = √(1-a^2)
  y = zeros(N)
  y[1] = rand(rng,Normal())
  for n in 2:N
    μ = a*y[n-1]
    y[n] = rand(rng,Normal(μ,σϵ))
  end
  return y  
end

function MCMCDiagnostics.potential_scale_reduction(θss::AbstractVector{<:NamedTuple}...)
  Tθ = eltype(first(θss))
  e1 = first(first(θss))
  Rhats = []
  for (n,T) in zip(fieldnames(Tθ),fieldtypes(Tθ))
    rh = if T <: Number
      chains = [getfield.(θs,n) for θs in θss]
      potential_scale_reduction(chains...)
    elseif T <: AbstractArray
      e1p = getfield(e1,n)
      rhc = similar(e1p)
      for i in CartesianIndices(e1p)
        chains = [getproperty.(θs,n)[i] for θs in θss]
        rhc[i] = potential_scale_reduction(chains...)
      end
      rhc
    else error("Unsupported type $T for field $n")
    end
    push!(Rhats,n=>rh)
  end
  return (;Rhats...)
end
