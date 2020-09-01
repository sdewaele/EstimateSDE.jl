"
Autoregrssive models
"
module ArModels

export ArModelStruct,ArModel,scalararmodel,modelstruct,modelorder,signaldim,estimate,
  isstationary,poly2par,par2poly,refcoef,predpar,ArPredparIter,randinit,
  powerspectrum_ar

using Distributions
using LinearAlgebra
using Random
using DocStringExtensions: SIGNATURES
"""Test for stationarity of partial autocorrelations
Throws an exception if non-stationary
"""
function isstationary(pc)
  p = size(pc,3)
  for i in 1:p
    _,s,_ = svd(pc[:,:,i])
    all(s.<1) || error("Non-stationary partial autocorrelations")
  end
  true
end

"convert polynomial (I -leading parameters) to prediction parameter"
poly2par(pol) = -pol[:,:,2:end]

"convert prediction parameter to  polynomial (I -leading parameters)"
par2poly(par) = cat(Matrix{Float64}(I,size(par,1),size(par,1)),-par,dims=3)

"Autoregressive model structure"
struct ArModelStruct
  p::Int64
end

"
Autoregressive model

TODO Add FIELDS
"
struct ArModel
  "partial autocorrelation"
  pc::Array{Float64,3}
  "covariance matrix"
  R0::Hermitian{Float64,Array{Float64,2}}
  function ArModel(pc,R0)
    dim = size(R0,1)
    size(pc,2)==size(R0,1)==dim || error("Dimension mismatch")
    isstationary(pc) || error("Non-stationary partial correlations PC")
    isposdef(R0) || error("R0 is not positive-definite")
    new(pc,R0)
  end
end

"""scalararmodel(pc,R0) -> ArModel

Scalar AR model
"""
function scalararmodel(pc,σ)
  pcv = reshape(pc,1,1,length(pc))
  a = Array{Float64,2}(1,1)
  a[1,1] = σ^2
  R0 = Hermitian(a)
  ArModel(pcv,R0)
end

"model structure"
modelstruct(m::ArModel) = ArModelStruct(size(m.pc,3))
modelorder(m::ArModel) = modelstruct(m::ArModel).p
signaldim(m::ArModel) = size(m.R0,1)

"""randinit(m::ArModel) -> y

Draw random initial values from stationary distribution
of a vector autoregressive process pc,R0
"""
function randinit(m::ArModel)
    p = modelorder(m); dim = signaldim(m)
    μ = zeros(dim)
    Σ = convert(Array,m.R0)
    y = zeros(p,dim)
    i = 1
    for (par,P) in ArPredparIter(m)
      y[i,:] = rand(MvNormal(μ,Σ))
      i += 1
      parf = par[:,:,:,1]
      μ = sum(parf[:,:,c]*y[i-c,:] for c in 1:i-1)
      Σ = convert(Array,P)
    end
    y
end

"""
$SIGNATURES
Generate a sample of n observations from the AR model m
"""
function Base.rand(m::ArModel,n,yi=nothing)
  if yi==nothing yi = randinit(m) end
  p = modelorder(m)
  dim = signaldim(m)
  yw = zeros(n+p,dim)
  yw[1:p,:] = yi
  par,P = predpar(m)
  parf = par[:,:,:,1]
  Σ = convert(Array,P[end,1])
  ϵ = rand(MvNormal(Σ),n)
  for i in p+1:n+p
    μ = sum(parf[:,:,c,1]*yw[i-c,:] for c in 1:p)
    yw[i,:] = μ+ϵ[:,i-p]
  end
  y = yw[p+1:end,:]
  yi = y[end-p+1:end,:]
  y,yi
end

"Reflection coefficients"
function refcoef(m::ArModel)
  pc = m.pc
  R0 = m.R0
  p = modelorder(m)
  dim = signaldim(m)
  rc = Array{Float64}(undef,dim,dim,p,2)
  P = Array{Hermitian{Float64}}(undef,p,2)
  Pc = fill(R0,2)
  # Imat = Matrix{Float64}(I,dim,dim)
  for i in 1:p
    pcc = pc[:,:,i]
    F,B = map(x->cholesky(x).U',Pc)
    rc[:,:,i,1] = F*pcc/B
    rc[:,:,i,2] = B*pcc'/F
    Pc[1] = (I-F*pcc*pcc'/F)*Pc[1] |> Hermitian
    Pc[2] = (I-B*pcc'*pcc/B)*Pc[2] |> Hermitian
    P[i,:] = map(copy,Pc)
  end
  rc,P
end

# predpar function equivalent to the last return value of
# the ArPredparIter iterator defined below
"""Prediction parameters -> par,P

See also: `ArPredparIter`
"""
function predpar(m::ArModel)
  rc,P = refcoef(m)
  p = modelorder(m)
  dim = signaldim(m)
  par = Array{Float64}(undef,dim,dim,p,2)
  par[:,:,1,:] = rc[:,:,1,:]
  parxrc = Array{Float64}(undef,dim,dim,p,2)
  for i in 2:p
    rcc = rc[:,:,i,:]
    for d in 1:2
      reverse_d = d==1 ? 2 : 1
      for ii in 1:i-1
        parxrc[:,:,ii,d] = rcc[:,:,d]*par[:,:,ii,reverse_d]
      end
    end
    par[:,:,1:i-1,:] = par[:,:,1:i-1,:]-parxrc[:,:,i-1:-1:1,:]
    par[:,:,i,:] = rcc
  end
  par,P
end

"""Iterator for prediction parameters of increasing order

See also: `predpar`
"""
struct ArPredparIter
  m::ArModel
  par::Array{Float64,4}
  parxrc::Array{Float64,4}
  rc::Array{Float64,4}
  P::Array{Hermitian{Float64},2}
  p::Int64
  function ArPredparIter(m)
    p = modelorder(m)
    dim = signaldim(m)
    par = Array{Float64}(undef,dim,dim,p,2)
    parxrc = Array{Float64}(undef,dim,dim,p,2)
    rc,P = refcoef(m)
    new(m,par,parxrc,rc,P,p)
  end
end
# Base.eltype(::Type{ArPredparIter}) = Tuple{Array{Float64,4},Hermitian{Float64,Array{Float64,2}}}
Base.length(pp::ArPredparIter) = pp.p
function Base.iterate(pp::ArPredparIter,state=1)
  i = state
  i≤pp.p || return nothing
  rcc = pp.rc[:,:,i,:]
  if i>1
    rcc = pp.rc[:,:,i,:]
    for d in 1:2
      reverse_d = d==1 ? 2 : 1
      for ii in 1:i-1
        pp.parxrc[:,:,ii,d] = rcc[:,:,d]*pp.par[:,:,ii,reverse_d]
      end
    end
    pp.par[:,:,1:i-1,:] = pp.par[:,:,1:i-1,:]-pp.parxrc[:,:,i-1:-1:1,:]
  end
  pp.par[:,:,i,:] = rcc
  parc = pp.par[:,:,1:i,:]
  (parc,pp.P[i]),i+1
end

"""
$SIGNATURES

Estimate AR model from N-by-dim data x
Uses the Nuttall-Strand algorithm
Reference: Marple, "Digital spectral analysis" p. 406, eq. 15.98
"""
function Distributions.estimate(m::ArModelStruct,x)
  N,dim = size(x)
  R0 = (x'*x)/N |> Hermitian
  rc = Array{Float64}(undef,dim,dim,m.p,2)
  pc = Array{Float64}(undef,dim,dim,m.p)
  P = Array{Hermitian}(undef,m.p,2)
  Pc = [R0,R0]
  f = b = x'
  for i in 1:m.p
    v = f[:,2:end]
    w = b[:,1:end-1]
    Rvv = (v*v')/N
    Rww = (w*w')/N
    Rvw = (v*w')/N
    Δ = sylvester(Rvv/Pc[1],Pc[2]\Rww,-2Rvw)
    F,B = map(x->cholesky(x).U',Pc)
    pcc = pc[:,:,i] = F\Δ/B'
    rcf = F*pcc/B
    rc[:,:,i,1] = rcf
    rcb = rc[:,:,i,2] = B*pcc'/F
    f = v-rcf*w; b = w-rcb*v
    Pc[1] = (I-F*pcc*pcc'/F)*Pc[1] |> Hermitian
    Pc[2] = (I-B*pcc'*pcc/B)*Pc[2] |> Hermitian
    P[i,:] = map(copy,Pc)
  end
  yi = x[end-m.p+1:end,:]
  ArModel(pc,R0),P,yi
end

"
$SIGNATURES

Power spectrum for a scalar autoregressive (AR) model with prediction
parameters `a` and generating noise std `σϵ` estimated from a signal
sampled at interval `T` at frequencies `f`.
"
function powerspectrum_ar(a,σϵ,T,f)
  den = ones(ComplexF64,size(f))
  for (k,ac) in enumerate(a)
    @. den -= ac*exp(-k*1im*2π*f*T)
  end
  h_ar = @. T*σϵ^2/abs(den)^2
  return h_ar
end

end
