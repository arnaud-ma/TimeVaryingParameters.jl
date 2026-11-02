"""
    TimeVaryingParameters

Statistical models whose parameters vary smoothly with time.

Each parameter is a basis expansion pushed through a link,

    θⱼ(t) = linkⱼ⁻¹( Bⱼ(t) · βⱼ )

so a model is a [`ParamSpec`](@ref) -- one basis and one link per named parameter -- and,
with a distribution constructor, a [`TimeVaryingFamily`](@ref) whose value at a time is an
ordinary `Distributions.jl` object.
"""
module TimeVaryingParameters

using ArgCheck: @argcheck
using LinkFunctions: LinkFunctions, Link, linkfun, linkinv
using LogExpFunctions: logistic, logit
using Distributions: Distributions, logpdf
using ADTypes: ADTypes
using OptimizationBase: OptimizationFunction, OptimizationProblem, solve
using SciMLBase: SciMLBase
using StatsAPI: StatsAPI, fit, coef, params

# Links
export IntervalLink

# Bases
export AbstractBasis, ConstantBasis, PolynomialBasis, FourierBasis, IndicatorBasis
export CombinedBasis, TrigBasis
export nbasis, eval_basis, eval_basis!, design_matrix

# Parametrization
export ParamSpec, paramnames, ncoef, coefs, parameters_at, initial_coefficients
export blockindex, blockrange, apply_link, invert_link

# Families and objectives
export TimeVaryingFamily, DistConstructor, PeriodicDesign, group_by_time, distributions, linear_predictors
export PeriodicNLL, PointwiseNLL, PointwiseLoss

# Fitting
export TimeVaryingFit, solve_objective, curve
export fit, coef, params

include("links.jl")
include("bases.jl")
include("spec.jl")
include("family.jl")
include("fit.jl")

end # module
