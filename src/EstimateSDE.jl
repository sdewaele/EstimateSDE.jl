"""
Estimate Stochastic Differential Equation

This module uses @debug to report estimation error messages.
"""
module EstimateSDE

export isdiagonalizable,TimeSeriesModel,VectorSde,
        statedim,measdim,logpdf,testprocess,
        autocovz,autocov,statcovz,statcov,randtimes, convert
export Sde
export checkstable, isstable, isvalid
export posterior_samples
export logpdf_reg_refprior
export logpdf2pdf
export describe_model
export powerspectrum
export invscode
export unique_index
export modelvariation
export MlEstimator
export Problem,ProblemProb,ProblemLike, transformation
export PmeanEstimator, State, State, statstate
export SdeProcess, testprocess, SdeProcess, order, kldiv, WhiteNoise
export detstate, lowerorder, powerspectrum_ar
export ArModelStruct, predpar, interpolate,initial_estimate
export SdeRoots,diag_vectorsde
export SdeEngine,SdeRootsEngine,ElimQ,MatFrac,VectorSdeEngine
export VectorSdeProcess,logpdf_ml_σϵ
export logpdf_state, paramstuple
export transformation, transform_ss, sdetype
export AddRoot, InitialModel, ResampleAr
export sortroots, logpdf_ml_white_noise
export Obs, semiposdef_mvrand, combineroots, splitroots
export sde,ml_estimate_init, engine
export parint, AddComplexRoot,RemoveRoots,default_init_tors
export ScalarSde
export estimate,mean,rand
export SquaredExp,defaultengine, datacov, ReferencePrior
export std, EstimateInfo, EstimationStage, label
export systemstatus, tracetable, reporttrace
export issorted_roots, issorted_rootstuple
export in
export safe_kldiv
export maxlike_est, bringin
export BatchDataSettings,batch_data
export BatchMlEstimation,batch_mle_trace_single,batch_mle_trace
export default_random_init_tors
export batchsettings
export batch_posterior_samples
export trace_estimates
export settings_resultsdir
export batch_model_evaluation
export derived_estimates
export copy_icmla_figures

using Distributions
using Optim
using Polynomials
using Zygote
using Parameters
using TransformVariables
using LogDensityProblems
using LogDensityProblems: logdensity_and_gradient
using DynamicHMC
using DocStringExtensions: TYPEDEF, FIELDS, SIGNATURES
using ToeplitzMatrices
using LinearAlgebra
using Random
using Printf
using DataStructures
using Requires
using .Threads
using DSP
using QuickTypes
using Statistics
using StaticArrays
using Intervals
using MCMCDiagnostics
using Plots
using Dates
using InteractiveUtils
using Serialization
using DataFrames
using Pipe
using ProgressMeter
using Pkg.TOML
using REPL.TerminalMenus

basedir = abspath(joinpath(@__DIR__,".."))

"Results directory"
function resultsdir(parts...)
  d = joinpath(homedir(),"results","EstimateSDE",parts...)
  if !ispath(d) mkpath(d) end
  return d
end

function __init__()
  @require MD5="6ac74813-4b46-53a4-afec-0b5dc9d7885c" begin
      @require PyCall="438e738f-606a-5dbb-bf0a-cddfbfd45ab0" begin 
        include("stan.jl")
      end
  end
end

include("armodels.jl")
using .ArModels
include("utilities.jl")
include("state.jl")
include("vector-sde.jl")
include("scalar-sde.jl")
include("white-noise.jl")
include("squared-exp.jl")
include("roots.jl")
include("likelihood.jl")
include("problem.jl")
include("random.jl")
include("transformations.jl")
include("estimators-init.jl")
include("estimators.jl")
include("kullback-leibler.jl")
include("report.jl")
include("batch.jl")

end # module
