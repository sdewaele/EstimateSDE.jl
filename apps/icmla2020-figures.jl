include("../batch/figure-whitenoise.jl")

include("../batch/figure-mle-accuracy.jl")

include("../batch/figure-cmp-estimators.jl")

include("../batch/figure-fit-kld.jl")

choice = copy_icmla_figures(figure_dir)

include("figure-rainfall.jl")

copy_icmla_figures(results_dir,choice)
nothing