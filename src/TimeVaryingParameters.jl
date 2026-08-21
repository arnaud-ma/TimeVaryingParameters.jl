"""
    TimeVaryingParameters

A generalized framework for Time-Varying M-Estimation via Basis Expansion.
"""
module TimeVaryingParameters

using ArgCheck: @argcheck
using LinkFunctions: LinkFunctions
using OptimizationBase: OptimizationFunction, OptimizationProblem, solve
using ADTypes: ADTypes
using SciMLBase: SciMLBase
using StatsAPI: StatsAPI, fit, params
using LinearAlgebra: hcat, dot
using ComponentArrays: ComponentVector

export fit, params
export AbstractBasis, ConstantBasis, PolynomialBasis, FourierBasis, IndicatorBasis
export TimeVaryingModel

public AbstractFitted


# ──────────────────────────────────────────────────────────────────────────── #
# Core abstractions                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

abstract type AbstractBasis end
abstract type AbstractFitted end

Base.broadcastable(x::AbstractBasis) = Ref(x)
Base.broadcastable(x::AbstractFitted) = Ref(x)

# ──────────────────────────────────────────────────────────────────────────── #
# Basis definitions                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

"""
    ConstantBasis <: AbstractBasis

Represents an intercept. Yields a column of ones.
"""
struct ConstantBasis <: AbstractBasis end
eval_basis(::ConstantBasis, ts::AbstractVector) = ones(Float64, length(ts), 1)
eval_basis(::ConstantBasis, _t::Real) = [1.0]

"""
    PolynomialBasis <: AbstractBasis

A polynomial trend of a given degree: t, t^2, ..., t^d.
Note: Does not include the intercept. Add ConstantBasis() for an intercept.
"""
struct PolynomialBasis <: AbstractBasis
    degree::Int
end
function eval_basis(b::PolynomialBasis, ts::AbstractVector)
    return reduce(hcat, [ts .^ d for d in 1:b.degree])
end
eval_basis(b::PolynomialBasis, t::Real) = [t^d for d in 1:b.degree]

"""
    FourierBasis{T<:Real} <: AbstractBasis

A purely trigonometric basis of a given order and period.
Note: Does not include the intercept baseline.
"""
struct FourierBasis{T <: Real} <: AbstractBasis
    length_period::T
    order::Int
end
function eval_basis(b::FourierBasis, ts::AbstractVector)
    ω = 2 * π / b.length_period
    cols = Vector{Vector{Float64}}(undef, 2 * b.order)
    for k in 1:b.order
        cols[2k - 1] = cos.(k .* ω .* ts)
        cols[2k] = sin.(k .* ω .* ts)
    end
    return reduce(hcat, cols)
end
function eval_basis(b::FourierBasis, t::Real)
    ω = 2 * π / b.length_period
    vals = zeros(Float64, 2 * b.order)
    for k in 1:b.order
        vals[2k - 1] = cos(k * ω * t)
        vals[2k] = sin(k * ω * t)
    end
    return vals
end

"""
    IndicatorBasis{F} <: AbstractBasis

Splits time into distinct groups (e.g., seasons, days).
Acts as a one-hot encoded design matrix.
"""
struct IndicatorBasis{F, L <: AbstractVector} <: AbstractBasis
    split_by::F
    levels::L # Discovered or pre-defined levels
end
IndicatorBasis(split_by) = IndicatorBasis(split_by, Any[])

function eval_basis(b::IndicatorBasis, ts::AbstractVector)
    groups = b.split_by.(ts)
    # If levels aren't defined, discover them
    levels = isempty(b.levels) ? unique(groups) : b.levels

    mat = zeros(Float64, length(ts), length(levels))
    for (j, lvl) in enumerate(levels)
        mat[:, j] .= (groups .== lvl)
    end
    return mat
end
function eval_basis(b::IndicatorBasis, t)
    @argcheck !isempty(b.levels) "IndicatorBasis levels must be known to evaluate single points."
    group = b.split_by(t)
    return Float64[group == lvl for lvl in b.levels]
end

# ──────────────────────────────────────────────────────────────────────────── #
# Basis arithmetic                                                             #
# ──────────────────────────────────────────────────────────────────────────── #
struct CombinedBasis{T <: Tuple} <: AbstractBasis
    bases::T
end

Base.:+(b1::AbstractBasis, b2::AbstractBasis) = CombinedBasis((b1, b2))
Base.:+(b1::CombinedBasis, b2::AbstractBasis) = CombinedBasis((b1.bases..., b2))
Base.:+(b1::AbstractBasis, b2::CombinedBasis) = CombinedBasis((b1, b2.bases...))
Base.:+(b1::CombinedBasis, b2::CombinedBasis) = CombinedBasis((b1.bases..., b2.bases...))

function eval_basis(b::CombinedBasis, ts::Union{AbstractVector, Real})
    mats = map(basis -> eval_basis(basis, ts), b.bases)
    return ts isa Real ? vcat(mats...) : hcat(mats...)
end

# ──────────────────────────────────────────────────────────────────────────── #
# Optimizer defaults (boilerplate)                                             #
# ──────────────────────────────────────────────────────────────────────────── #

const DEFAULT_OPTIMIZER = Ref{Any}(nothing)
function default_optimizer()
    if isnothing(DEFAULT_OPTIMIZER[])
        throw(ArgumentError("No optimizer detected. Set default or pass explicitly."))
    end
    return DEFAULT_OPTIMIZER[]
end

const DEFAULT_OPTIMS_KWARGS = (;
    adtype = ADTypes.AutoForwardDiff(),
    optim_function_kwargs = (;),
    optim_problem_kwargs = (;),
    optim_solve_kwargs = (;),
)

function get_optim_func(score_func, init; optim = default_optimizer(), kwargs_...)
    k = merge(DEFAULT_OPTIMS_KWARGS, kwargs_)
    function optim_func(null_data = nothing)
        f = OptimizationFunction(score_func, k.adtype; k.optim_function_kwargs...)
        prob = OptimizationProblem(f, init, null_data; k.optim_problem_kwargs...)
        return solve(prob, optim; k.optim_solve_kwargs...)
    end
    return optim_func
end

# ──────────────────────────────────────────────────────────────────────────── #
# Objective function & model definition                                        #
# ──────────────────────────────────────────────────────────────────────────── #
struct TimeVaryingModel{B <: NamedTuple, L <: NamedTuple}
    bases::B # A Tuple of AbstractBasis, one for each parameter
    links::L
end
TimeVaryingModel(; bases, links) = TimeVaryingModel(bases, links)

struct TimeVaryingObjective{F, M, L}
    func::F
    design_matrices::M # NamedTuple of precomputed matrices B_j
    link_funcs::L      # NamedTuple of link functions
    xs::Vector{Float64}
end

function (obj::TimeVaryingObjective)(θ, _null_data)
    names = keys(obj.design_matrices)
    total = zero(eltype(θ))

    for i in eachindex(obj.xs)
        params = eval_params_at(obj.design_matrices, obj.link_funcs, θ, i)
        total += obj.func(Tuple(params), obj.xs[i])
    end
    return total
end

@inline function eval_params_at(design_matrices::NamedTuple{names}, link_funcs::NamedTuple{names}, θ, i) where {names}
    return _eval_params_at(design_matrices, link_funcs, θ, i, Val(names))
end

@inline _eval_params_at(design_matrices, link_funcs, θ, i, ::Val{()}) = ()

@inline function _eval_params_at(design_matrices, link_funcs, θ, i, ::Val{names}) where {names}
    name = first(names)
    B = getfield(design_matrices, name)   # NamedTuple -> getfield is fine
    θj = getproperty(θ, name)              # ComponentVector -> needs getproperty
    η = zero(eltype(θ))
    @inbounds for k in axes(B, 2)
        η += B[i, k] * θj[k]
    end
    val = LinkFunctions.linkinv(getfield(link_funcs, name), η)
    return (val, _eval_params_at(design_matrices, link_funcs, θ, i, Val(Base.tail(names)))...)
end
# ──────────────────────────────────────────────────────────────────────────── #
# Fitting                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #
struct TimeVaryingFitted{M, R, P, L, B} <: AbstractFitted
    model::M
    optim_res::R
    θ_flat::P
    link_funcs::L
    bases::B # Store the exact bases (important for IndicatorBasis discovery)
end

"""
    StatsAPI.fit(model::TimeVaryingModel, func, xs, ts; init, kwargs...)
"""
function StatsAPI.fit(
        model::TimeVaryingModel,
        func, xs::AbstractVector, ts::AbstractVector;
        init,
        warn_unsuccessful = true,
        optim_kwargs...
    )
    @argcheck length(ts) == length(xs)
    @argcheck keys(init) == keys(model.links)

    names = keys(model.bases)
    @argcheck keys(init) == names

    design_matrices = map(b -> eval_basis(b, ts), model.bases)

    init_blocks = map(names) do name
        θ_guess = init[name]
        link = model.links[name]
        ncols = size(design_matrices[name], 2)
        if θ_guess isa Real
            v = zeros(ncols)
            v[1] = LinkFunctions.linkfun(link, θ_guess)
            Float64.(v)
        else
            @argcheck length(θ_guess) == ncols
            Float64.(θ_guess)
        end
    end
    init_all = ComponentVector(NamedTuple{names}(init_blocks))

    objective = TimeVaryingObjective(func, design_matrices, model.links, Float64.(xs))
    optim_func = get_optim_func(objective, init_all; optim_kwargs...)
    res = optim_func() # No data passed here, it is captured in the closure

    if warn_unsuccessful && !SciMLBase.successful_retcode(res.retcode)
        @warn "Optimization failed for Time-Varying fitting."
    end

    # Return fitted model, explicitly preserving the bases (with discovered levels)
    return TimeVaryingFitted(model, res, res.u, model.links, model.bases)
end

# ──────────────────────────────────────────────────────────────────────────── #
# Evaluation                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

StatsAPI.params(fitted::TimeVaryingFitted) = fitted.θ_flat

function StatsAPI.params(fitted::TimeVaryingFitted, t)
    names = keys(fitted.bases)
    return map(names) do name
        b_t = eval_basis(fitted.bases[name], t)
        η = dot(b_t, fitted.θ_flat[name])
        LinkFunctions.linkinv(fitted.link_funcs[name], η)
    end
end

end # module TimeVaryingParameters
