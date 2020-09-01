using Test
using EstimateSDE
using Distributions
using LinearAlgebra
using Random
using Zygote

@testset "EstimateSDE" begin
ty =[ 
  0.69   4.6   5.72   8.09   8.19   8.21  10.11  10.37  11.88  12.46
 -0.31  -0.02  0.12  -0.23  -0.24  -0.24   0.15   0.24   0.55   0.47
]
data = map(x->Obs(x...),eachcol(ty))
a = [0.371, 0.0693, 1.26, 0.124]
eq = Sde(a,0.05)
fixed_state = State([data[1].y,0.2,-0.05,2.1])

@testset "Kalman equations" begin
  μ = [5.99,0.45,3.49,0.23]
  Σ = [
     5.71  -1.11  -3.71  -0.49
    -1.11   2.58   0.26  -1.94
    -3.71   0.26   2.84   0.36
    -0.49  -1.94   0.36  11.48
  ]
  proc = SdeProcess(eq,State(μ,Σ))
  Δt = 2.3
  for engine in EstimateSDE.ENGINES
    proc′ = transform_ss(sdetype(engine),proc)
    eq′ = proc′.eq
    state′ = proc′.state
    e = engine(eq)
  
    μy,σ2y = EstimateSDE.obsdist(e,state′)
    @test μy≈5.99
    @test σ2y≈5.71
  
    state_m′ = EstimateSDE.measure(e,state′,-4.3,μy,σ2y)
    state_m = transform_ss(Sde,SdeProcess(eq′,state_m′)).state
    @test state_m.μ≈[-4.299999999999999, 2.450332749562172, 10.175796847635727, 1.1130297723292468]
    @test state_m.Σ≈[-8.881784197001252e-16 0.0 0.0 0.0; 0.0 2.3642206654991242 -0.4612084063047286 -2.0352539404553416; 0.0 -0.4612084063047286 0.4294746059544656 0.04162872154115588; 0.0 -2.0352539404553416 0.04162872154115588 11.437950963222418]
  
    state_p′ = EstimateSDE.predict(e,state′,Δt)
    state_p = transform_ss(Sde,SdeProcess(eq′,state_p′)).state
    @test state_p.μ≈[ 10.30350225480549,-1.0284495977990458,-6.621680617195209,-4.072984824995861]
    @test state_p.Σ≈[21.053696357059337 19.2706890985522 0.7602965515146454 -15.198802594403409; 19.2706890985522 21.706321456728876 2.464682525360381 -17.17083607634992; 0.7602965515146454 2.464682525360381 2.365832518117748 -2.1570291778727433; -15.198802594403409 -17.17083607634992 -2.1570291778727433 13.789300684647449]
  end
end

@testset "logpdf" begin
  logpdf(SdeProcess(eq),data)≈-25538.711601624374
  logpdf(SdeProcess(eq,fixed_state),data)≈-29370.38457270086
  lps_ref = logpdf(SdeProcess(eq),data)
  lpf_ref = logpdf(SdeProcess(eq,fixed_state),data)
  proc = SdeProcess(eq,fixed_state)
  for E in EstimateSDE.ENGINES
    e = E(eq)
    lps,_ = logpdf_state(e,statstate(e),data)
    @test lps≈lps_ref atol=0.1
    state_fixed′ = transform_ss(sdetype(E),proc).state
    lpf,_ = logpdf_state(e,state_fixed′,data)
    @test lpf≈lpf_ref atol=0.1
  end
end

@testset "SDE roots" begin
  eqr = testprocess(:C).eq
  eq = Sde(eqr)
  r = eqr.r
  p = order(eqr)
  U = [rc^(j-1) for j in 1:p, rc in r]
  g = zeros(p); g[end] = 1
  v = U\g
  Q′ = eqr.σϵ^2*v*v'
  Σs′ = [-Q′[i,j]/(r[i]+conj(r[j])) for i in 1:p,j in 1:p]
  A′ = Diagonal(r)
  # Continuous Lyapunov equation
  @test A′*Σs′+Σs′*A′'+Q′≈zeros(p,p) atol=1e-14
  # Compare to diagonalized vector SDE(1) process
  veq = diag_vectorsde(convert(VectorSde,eq))
  @test A′≈veq.A
  @test Q′≈veq.Q
  @test Σs′≈statcovz(veq)      
end

@testset "logpdf_ml_σϵ" begin
  ty =[ 
    0.69   4.6   5.72   8.09   8.19   8.21  10.11  10.37  11.88  12.46
  -0.31  -0.02  0.12  -0.23  -0.24  -0.24   0.15   0.24   0.55   0.47
  ]
  data = map(x->Obs(x...),eachcol(ty))
  a = [0.371, 0.0693, 1.26, 0.124]
  eq = Sde(a,0.05)
  fixed_state = State([data[1].y,0.2,-0.05,2.1])
  proc = SdeProcess(eq,fixed_state)
  eq1 = Sde(eq.a,1)
  for E in EstimateSDE.ENGINES
    e = E(eq1)
    for stationary in [true,false]
      state′ = if stationary
        statstate(eq1)
      else
        transform_ss(sdetype(E),proc).state
      end
      lp,σϵ_ml = logpdf_ml_σϵ(e,state′,data)
      σϵ_ml_test = stationary ? 3.575206949538863 : 4.041520808012871
      @test σϵ_ml≈σϵ_ml_test atol=0.05
    end
  end
end

@testset "transform variable" begin
  using TransformVariables
  eq = Sde([0.371, 0.0693, 1.26, 0.124],0.05)
  z = [0.3,1.1,-0.7,0.2]
  proc = SdeProcess(eq,State(z))
  data = rand(MersenneTwister(12202),proc,200,1.0)

  problem = ProblemLike(data;p=order(proc),engine=ElimQ)
  𝓣 = transformation(problem,proc)
  𝓣⁻¹ = inverse(𝓣)
  θ = paramstuple(problem,proc)
  θ_rec = (𝓣∘𝓣⁻¹)(θ)
  eq_rec = Sde(θ_rec,eq.σϵ)
  @test eq_rec.a≈eq.a atol=1e-10

  problem = ProblemLike(data;p=order(proc),engine=SdeRootsEngine)
  proc′ = transform_ss(problem,proc)
  𝓣 = transformation(problem,proc′)
  𝓣⁻¹ = inverse(𝓣)
  θ = paramstuple(problem,proc′)
  θ_rec = (𝓣∘𝓣⁻¹)(θ)
  eq′_rec = SdeRoots(θ_rec,eq.σϵ)
  @test eq′_rec.r≈proc′.eq.r atol=1e-8
end

@testset "initial estimates" begin
  proc = testprocess(:C)
  data = rand(MersenneTwister(12202),proc,200,1.0)
  problem = ProblemLike(data;p=order(proc)+2)
  Lwn,σ = logpdf_ml_white_noise(data,false)
  est_wn = WhiteNoise(σ)
  init_tors = [AddRoot();AddComplexRoot()]
  proc_prev = proc
  for m in init_tors
    esti = initial_estimate(m,problem,[proc_prev])
    L = logpdf(esti,problem.data)
    # NOTE: L>Lwn is not necessarily met for all signals and initial estimators
    # However, this condition IS met for the current test data and estimators.
    @test L>Lwn
  end  
end

@testset "autodiff" begin
  # The numerical gradient computed in `ngradient` can be very inaccurate because it is
  # affected by the rouding errors in the Kalman likelihood computation.
  function ngradient(f, xs::AbstractArray...)
    grads = zero.(xs)
    for (x, Δ) in zip(xs, grads), i in 1:length(x)
      δ = 1e-5 # sqrt(eps())=1e-8 is too small; results in large rounding errors
      tmp = x[i]
      x[i] = tmp - δ/2
      y1 = f(xs...)
      x[i] = tmp + δ/2
      y2 = f(xs...)
      x[i] = tmp
      Δ[i] = (y2-y1)/δ
    end
    return grads
  end
  proc = SdeProcess(testprocess(:C).eq)
  data = rand(MersenneTwister(12202),proc,200,1.0)

  for E in [ElimQ,MatFrac,SdeRootsEngine]
    problem = ProblemLike(data;p=order(proc),engine=E)
    proc′ = transform_ss(problem,proc)
    θ = paramstuple(problem,proc′)
    𝓣 = transformation(problem,proc′)
    η = inverse(𝓣,θ)
    Zygote.refresh()
    yf,B = Zygote.pullback(problem∘𝓣,η)
    dη = B(1)[1]
    ng = ngradient(problem∘𝓣,η)[1]
    @test dη≈ng atol=0.5
  end
end

end # @testset EstimateSDE begin