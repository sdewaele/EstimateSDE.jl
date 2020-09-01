# Roots of the characteristic equation of an SDE

"""
$SIGNATURES

Split roots into real and complex-valued roots. For the complex-valued roots,
only provides one of a complex conjugate pair. An error is raised if there is
no single complex conjugate
"""
function splitroots(r)
  rw = copy(r)
  rrc = empty(rw); ri = empty(rw)
  while !isempty(rw)
    rc = popfirst!(rw)
    if isreal(rc)
      push!(rrc,rc)
    else
      push!(ri,rc)
      is_conj = rw.==conj(rc)
      if count(is_conj)==0  error("No complex conjugate found for $rc")  end
      rw = rw[.!is_conj]
    end
  end
  rr = real(rrc)
  if 2length(ri)+length(rr)≠length(r)
    error("Order mismatch; complex roots may not occur in conjugate pairs")
  end
  return rr,ri
end

function splitroots(θ::NamedTuple)
  r_re = θ.r_re
  ri = if haskey(θ,:r_im)
    # zip not supported by Zygote; using buffer.
    # r_re_c = r_re[1:length(θ.r_im)]
    # map(x->complex(x...),zip(r_re_c,θ.r_im))
    nc = length(θ.r_im)
    B = zeros(complex(eltype(θ.r_im)),nc)
    ri_buf = Zygote.bufferfrom(B)
    for i in 1:nc
      ri_buf[i] = complex(r_re[i],θ.r_im[i])
    end
    copy(ri_buf)
  else
    (complex(eltype(r_re)))[]
  end
  nc = length(ri)
  rr = r_re[nc+1:end]
  return rr,ri
end

"
$SIGNATURES

Combine roots
"
function combineroots(rr,ri)
  if  !(eltype(rr)<:Real && eltype(ri)<:Complex) error("r_re must be real, and r_im complex")  end
  # ru = vcat(rr,ri,conj(ri)) # Zygote does not support
  nr = length(rr); ni = length(ri)
  ru_buf = Zygote.bufferfrom(zeros(eltype(ri),nr+2ni))
  ru_buf[1:nr] = rr
  ru_buf[nr.+(1:ni)] = ri
  ru_buf[nr.+ni.+(1:ni)] = conj(ri)
  ru = copy(ru_buf)
  r = sortroots(ru)
  return  r
end

sortroots(r) = sort(r;by=x->(real(x), imag(x)))

issorted_roots(r) = issorted(r;by=x->(real(x), imag(x)))

function issorted_rootstuple(θ)
  rr,ri = splitroots(θ)
  return issorted(rr) && issorted_roots(ri)
end

"""
$SIGNATURES

Random roots
"""
function randroots(rng,r,f,N,T)
  rreal,rcomplex = splitroots(r)
  Tr,Tc = eltype(rreal),eltype(rcomplex)
  rreald = map(rreal) do rc
    sdα = normparstd(rc*T)
    s = max(1,min(sdα,10)/T)
    σ = f*s/√N
    rd = uniform_rand(rng,rc,σ)
  end |> Vector{Tr}
  rcomplexd = map(rcomplex) do rc
    # Real component
    x = real(rc)
    sdα = normparstd(x*T)
    s = max(1,min(sdα,10)/T)
    σ = f*s/√N
    rdr = uniform_rand(rng,x,σ)
    # Imaginary component
    y = imag(rc)
    σ = f*1/T/√N
    rdi = uniform_rand(rng,y,σ)
    Complex(rdr,rdi)
  end |> Vector{Tc}
  rd = combineroots(rreald,rcomplexd)
  return rd
end

"""
$FIELDS

Roots representation of a Stochastic Differential Equation `Sde`.
"""
struct SdeRoots{T<:Complex,U} <: ScalarSde{T}
  "Roots"
  r::Vector{T}
  "Driving noise standard deviation"
  σϵ::U
  function SdeRoots(r::Vector{T},σϵ::U) where {T,U<:Real}
    if !(σϵ>0)  error("σϵ must be positive")  end
    new{T,U}(r,complex(σϵ))
  end
end

function Base.show(io::IO,eq::SdeRoots)
  p = order(eq)
  print(io,"SDE($p) roots "*join(rootstr.(eq.r),"; "))
  print(io,@sprintf " | σϵ = %0.3g" eq.σϵ)
end

function Base.show(io::IO,::MIME"text/plain",eq::SdeRoots)
  print(io,"Roots of Scalar Stochastic Differential Equation\n",eq)
end

function SdeRoots(θ::NamedTuple,σϵ)
  rr,ri = splitroots(θ)
  r = combineroots(rr,ri)
  return SdeRoots(r,σϵ)
end

function SdeRoots(r::AbstractVector{T};σy=one(real(T))) where {T}
  σϵ1 = one(real(T))
  g = std(SdeRoots(r,σϵ1))
  σϵ = σy/g
  return SdeRoots(r,σϵ)
end

function Statistics.std(eq::SdeRoots)
  if order(eq)==0 return eq.σϵ end
  mv = convert(VectorSde,eq)
  Σs = statcovz(mv)
  σy = sum(Σs) |> real |> √
  return σy
end

function SdeRoots(eq::Sde{T}) where {T}
  p = Polynomial([eq.a; one(T)])
  r = complex(roots(p))
  return SdeRoots(sortroots(r),eq.σϵ)
end

Polynomials.roots(eq::AbstractSde) = SdeRoots(eq).r

function Base.convert(::Type{VectorSde},eq::SdeRoots)
  r = eq.r
  if length(unique(r))<length(r) error("Roots must be unique") end
  p = order(eq)
  A = Diagonal(r)
  C = ones(eltype(r),1,p)
  U = [rc^(j-1) for j in 1:p, rc in r]
  v = U\[zeros(p-1);1]
  Q = eq.σϵ^2*v*v'
  R = zeros(eltype(r),1,1)
  veq = VectorSde(A,C,Q,R)
  return veq
end

function Base.convert(::Type{AbstractSde{U}},eq::SdeRoots{T}) where {T,U<:Complex}
  r = convert(Vector{U},eq.r)
  σϵ = convert(real(U),eq.σϵ)
  return SdeRoots(r,σϵ)
end

function Statistics.mean(eqs::Vector{<:SdeRoots})
  r = mean(getproperty.(eqs,:r))
  σϵ = mean(getproperty.(eqs,:σϵ))
  return SdeRoots(r,σϵ)
end

"""
Matrix of eigenvectors `U` of the `A` matrix for the SDE roots `eq`
"""
function eigenvecA(eq::SdeRoots)
  r = eq.r
  p = order(eq)
  # Next line not supported by Zygote, using Buffer instead
  # U = [rc^(i-1) for i in 1:p, rc in r]
  U_buf = Zygote.Buffer(r,p,p)
  for i in 1:p, (j,rc) in enumerate(r)
    U_buf[i,j] = rc^(i-1)
  end
  U = copy(U_buf)
  return U
end

function statcovz(eq::SdeRoots,U)
  p = order(eq)
  v = U\[zeros(p-1);1]
  Q = eq.σϵ^2*v*v'
  r = eq.r
  # Next line not supported by Zygote, using Buffer instead
  # Σs = [-Q[i,j]/(r[i]+conj(r[j])) for i in 1:p,j in 1:p]
  Σs_buf = Zygote.Buffer(v,p,p)
  for i in 1:p, j in 1:p
    Σs_buf[i,j] = -Q[i,j]/(r[i]+conj(r[j]))
  end
  Σs = copy(Σs_buf)
  return Σs
end

statcovz(eq::SdeRoots) = statcovz(eq,eigenvecA(eq))

"Complex to real conversion tolerance"
const def_tol = 1e-8

function roots2sdepar(r,tol=0)
  T = real(eltype(r))
  p = coeffs(fromroots(r))[1:end-1]
  a = map(p) do pc
    if !(abs(imag(pc))≤tol)
      error("Imaginary part exceeds tolerance, abs(",imag(pc),") > ",tol)
    end
    T(real(pc))
  end
  return a
end

function Sde(eq::SdeRoots,tol=def_tol)
  a = roots2sdepar(eq.r,tol)
  return Sde(a,eq.σϵ)
end

SdeRoots(eq::SdeRoots) = eq

SdeRoots(m::WhiteNoise{T}) where {T} = SdeRoots(complex(T)[],m.σ)

SdeRoots(proc::SdeProcess) = SdeRoots(proc.eq)

order(eq::SdeRoots) = length(eq.r)

"print roots"
function printroots(r)
  rstr = map(r) do rr
    s = @sprintf "%0.3g" real(rr)
    if !isreal(rr)
      s *= @sprintf " ± 2π%0.3gim" abs(imag(rr))/2π
    end
    s
  end
  w = maximum(length,rstr)
  for (i,rr) in enumerate(r)
    any(conj(rr).==r[1:i-1]) && continue
    rc = rstr[i]
    p = repeat(' ',w-length(rc))
    irc = @sprintf "-1/%0.3g" -1/real(rr)
    if !isreal(rr)
      irc *= @sprintf " ± (2π/%0.3g)im" 2π/abs(imag(rr))
    end
    s = "$rc$p = $irc"
    if real(rr)>0
      printstyled(s,"\n",color=:yellow)
    else
      println(s)
    end
  end
end

"compact root string"
function rootstr(x)
  return if isreal(x)
    @sprintf "%0.3g" real(x)
  else
    @sprintf "%0.3g%+0.3gim" real(x) imag(x)
  end
end

"
$SIGNATURES

Truncate real part of roots to `-R`.
"
function truncate_roots(r,R;warn=true)
  c = 0
  rt = map(r) do x
    a = real(x)
    return if a<-R
      c+=1
      isreal(r) ? -R : Complex(-R,imag(x))
    else
      x
    end
  end
  if c>0 && warn
    printroots(r)
    @warn "Truncated $c roots"
  end
  return rt
end
