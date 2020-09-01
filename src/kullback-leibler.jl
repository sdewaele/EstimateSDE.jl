## Kullback-Leibler divergence

function datacov_generic(proc,t)
  Δt1 = t[2]-t[1]
  regularly_sampled = all(diff(t).==Δt1)
  Σ = if regularly_sampled
    N = length(t)
    τ = (0:N-1)*Δt1
    R = autocov(proc,τ)
    SymmetricToeplitz(R)
  else
    Δt = [abs(t2-t1) for t1 in t,t2 in t]
    τ = sort(unique(Δt))
    τ_idx = indexin(Δt,τ)
    R = autocov(proc,τ)
    Hermitian(R[τ_idx])
  end
  return Σ
end

"""
$SIGNATURES

Data covariance matrix `Σ` for SDE process `proc` sampled at times `t`.
"""
datacov(proc,t) = datacov_generic(proc,t)

function datacov(proc::ScalarSdeProcess,t)
  if !has_statstate(proc;rtol=1e-4) @warn "Ignoring non-stationary state" end
  return datacov_generic(proc,t)
end

datacov(proc::WhiteNoise,t) = proc.σ^2*I(length(t))

"""
$SIGNATURES

Kullback-Leibler divergence of SDE process estimate `fh` with respect to the
true SDE process `f` for a time series observed at times `t`.
The current covariance-matrix based implementation has two disadvantages:

* It has a high computational load for a large dataset, say `length(t)>1000` 

* Both `f` and `fh` must be stationary.
"""
function kldiv(f,fh,t)
  N = length(t)
  Σ = datacov(f,t)
  Σh = datacov(fh,t)
  # eq:kld-covmat in continuous_discrete_bayesian_filtering_linear.lyx
  D = 1/2*(-(logdet(Σ)-logdet(Σh))+tr(Σh\Σ)-N)
  return D
end

safe_kldiv(f,fh,t,error_value=NaN) = try
  kldiv(f,fh,t)
catch e
  if !(e isa DomainError) rethrow() end
  @warn "kldiv thows error" f fh t
  error_value
end

