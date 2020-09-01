# White noise
"
White noise process

$FIELDS
"
struct WhiteNoise{T} <: TimeSeriesModel{T}
  "Standard deviation"
  σ::T
end

WhiteNoise() = WhiteNoise(1.0)

function Base.show(io::IO,wn::WhiteNoise)
  σstr = @sprintf "%0.3g" wn.σ
  print(io,"y = ϵ | σ = $σstr")
end

function Base.show(io::IO,::MIME"text/plain",wn::WhiteNoise)
  print(io,"Scalar white noise\n",wn)
end

order(m::WhiteNoise) = 0

Distributions.std(x::WhiteNoise) = x.σ

"
$SIGNATURES

log pdf of white noise model.
If `has_z` is `true`, the first observations is ignored, as is done for SDE(p)
processes in `logpdf` when `z` is provided. In this way, closer agreement
with the log probability of an SDE(p) process is achieved.
"
function Distributions.logpdf(proc::WhiteNoise,data,has_z=false)
  σ2 = proc.σ^2
  istart = has_z ? 2 : 1
  y = [x.y for x in data]
  Nl = length(y)
  lp = -1/2*(sum(y.^2)/σ2+Nl*log(σ2)+Nl*log(2π))
  return lp
end

function Base.rand(rng::AbstractRNG,proc::WhiteNoise,t::AbstractVector)
  data = map(x->Obs(x,proc.σ*randn(rng)),t)
  return data
end
