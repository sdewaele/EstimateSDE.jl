# Analyze Monsoon rainfail data from
# @Article{sinha2015trends,
#   Title                    = {Trends and oscillations in the Indian summer monsoon rainfall over the last two millennia},
#   Author                   = {Sinha, A. and Kathayat, G. and Cheng, H. and others},
#   Journal                  = {Nature communications},
#   Year                     = {2015},
#   Volume                   = {6},
#   Publisher                = {Nature Publishing Group},
# }
using EstimateSDE
using XLSX
using DataFrames
using Pipe:@pipe
using Plots
using Serialization
using Base.Threads
using Random
using GLM
using Parameters

## Settings
settings = (
  Tkld=1, # evalation time interval
  Tr=1.0, # Regular resampling time interval
  Tsub=5.0, # Irregular subsampling time interval
  nsub=5 # Number of irregularly subsampled signals
)
@unpack Tkld,Tr,Tsub,nsub = settings
ssize=(300,200)

## Rainfall data
data_dir = joinpath(pkgdir(EstimateSDE),"data")
fn = joinpath(pkgdir(EstimateSDE),"data","ncomms7309-s3.xlsx")
df = @pipe(
  DataFrame(XLSX.readtable(fn,1)...) |>
  rename(_,"Age Yr."=>"age") |>
  Float64.(_) |>
  sort(_,:age)
)

## Basic analysis
if nthreads()<6 @warn("Only $(nthreads()) threads. More will accelerate processing") end
t = df.age
Tav = mean(diff(t))
N = length(t)
println("N = $N observations; average sampling interval Tav = ",round(Tav;digits=2)," years")

## Fit trend
trendtype = :moving_average
df.O18trend = if trendtype==:moving_average
  tr = df.age[1]:Tr:df.age[end]
  yr = interpolate(df.age,df.O18,tr;method=:linear)
  win = 25 #10
  Nr = length(yr)
  yma = map(1:Nr) do i
    i1 = max(1,i-win)
    i2 = min(Nr,i+win)
    mean(yr[i1:i2])
  end
  interpolate(tr,yma,df.age;method=:linear,extra=true)
elseif trendtype==:singularspectrum
  L = 50 # Window length
  tr = df.age[1]:Tr:df.age[end]
  yr = interpolate(df.age,df.O18,tr;method=:linear)
  yt,ys = analyze(yr,L)
  interpolate(tr,yt,df.age;method=:linear,extra=true)
elseif trendtype==:mean
  fill(mean(df.O18),nrow(df))
end

df.O18detrend = df.O18-df.O18trend

## Plot data
plt_orig = plot(df.age,df.O18,xlabel="time (years)",ylabel="O18",label=nothing)
plot!(df.age,df.O18trend,label="trend",fg_legend=nothing,bg_legend=nothing)
plt_detrend = plot(df.age,df.O18detrend,xlabel="time (years)",ylabel="O18",label=nothing)

plt = plot(plt_orig,plt_detrend;layout=(2,1),size=2 .* ssize)
display(plt)

## Estimation setup
y = df.O18detrend
y .-= mean(y)
data = Obs.(t,y)
tor = MlEstimator()
Ts = [0.5,1,2,5,10]
init_tors = default_init_tors(MersenneTwister(12320),Ts)

models = []

## No subsampling
problem = ProblemLike(data;p=8)
proc_est,trace = @time estimate(tor,problem;init_tors=init_tors,ret_trace=true)
push!(models,(label="original",proc=proc_est,trace=trace,Tav=Tav,data=data))
println("Estimate original data: ",proc_est)

## Subsampled
rng = MersenneTwister(23493)
for isub in 1:nsub
  iirand = rand(rng,length(data)).<Tav/Tsub
  data_sub = data[iirand]
  Tav_sub = mean(diff(getfield.(data_sub,:t)))
  problem_sub = ProblemLike(data_sub;p=8)
  proc_est_sub,trace_sub = @time estimate(tor,problem_sub;init_tors=init_tors,ret_trace=true)
  push!(models,(label="sub $isub",proc=proc_est_sub,trace=trace_sub,Tav=Tav_sub,data=data_sub))
  println("isub = $isub estimate ",proc_est_sub)
end

## Save results
results_dir = EstimateSDE.resultsdir("rainfall-data")
f = joinpath(results_dir,"rainfall-sde-estimate.jld")
R = (settings=settings,models=models,df=df)
serialize(f,R); @info "Results saved to: $f"
