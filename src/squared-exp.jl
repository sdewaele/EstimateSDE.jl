# Squared exponential

"
Process with a squared exponential covariance function, with added
independent white noise.

$FIELDS
"
struct SquaredExp{T} <: TimeSeriesModel{T}
  "Time (/length) scale"
  l::T
  "Standard deviation,excluding white noise"
  σ::T
  "White noise standard deviation"
  σw::T
end

Base.show(io::IO,x::SquaredExp) = print(io,@sprintf("SquaredExp l = %0.1f σ = %0.1f σw = %0.2g",x.σ,x.l,x.σw))

autocov(proc::SquaredExp,τ) = proc.σ^2*exp.(-τ.^2 ./2proc.l^2)+autocov(WhiteNoise(proc.σw),τ)

# NOTE: Since white noise has a finite power that is spread over an infinite frequency range,
# it does not contribute to the continous power spectrum.
powerspectrum(proc::SquaredExp,f) = proc.σ^2*√(2π)*proc.l*exp.(-2(π*proc.l.*f).^2).+powerspectrum(WhiteNoise(proc.σw),f)