# Compare Covariance matrix computation with ODE solution
# Requires DifferentialEquations
# NOTE: Not included in regular test because DifferentialEquations
# is a heavy dependency, which will extend the duration of package testing

using DifferentialEquations
using EstimateSDE
using EstimateSDE:vectorprocess,kalman_predict,logpdf_precomp
using Random
using Parameters
using LinearAlgebra
using DataStructures
using Distributions

## Get covariance matrix Σ
proc = testprocess(:D)
N = 20
rng = MersenneTwister(3248342)

## Generate a starting state, yielding starting Σ
t,y = rand(rng,proc,N)
_,state = logpdf(proc,t,y;retstate=Val(true))

## Parameters for Pp
p = order(proc)
Δt = 1.3
D = logpdf_precomp(Val(:elimQ),proc.eq)
state_pred = kalman_predict(D,state,Δt)
Σ_pred = cov(state_pred)

## Numerica ODE solve
# dP/dt = AP+PA' + Q
vproc = vectorprocess(proc)
veq = vproc.eq
A,Q = veq.A,veq.Q
function sdecov!(du,u,p,t)
  du[:,:] = A*u+u*A'+Q
end
u0 = state.Σ
tspan = (0.0,Δt)
prob = ODEProblem(sdecov!,u0,tspan)
sol = solve(prob,abstol=1e-8,reltol=1e-6)
Σ_pred_ode = sol.u[end]
@show Σ_pred_ode≈Σ_pred
