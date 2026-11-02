# # Bases
#
# A basis turns a time `t` into a row of features. `design_matrix` stacks one row per
# observation, so the matrix is **time-major** and `B * β` is a single gemv -- the shape
# `SmoothPeriodicStatsModels.trig_basis` already produces.

abstract type AbstractBasis end

Base.broadcastable(b::AbstractBasis) = Ref(b)

"""
    nbasis(b::AbstractBasis) -> Int

Number of columns `b` contributes to a design matrix.
"""
function nbasis end

"""
    eval_basis!(out, b::AbstractBasis, t) -> out

Write the `nbasis(b)` features of `b` at time `t` into `out`.
"""
function eval_basis! end

# The numeric type a basis carries in its own fields (a period, a centring offset), so that
# `design_matrix` promotes rather than forcing `Float64`. `Bool` is the neutral element of
# `promote_type`, so a basis with no numeric field contributes nothing.
paramtype(::AbstractBasis) = Bool
basis_eltype(b::AbstractBasis, ::Type{T}) where {T} = float(promote_type(T, paramtype(b)))

"""
    eval_basis(b::AbstractBasis, t) -> Vector

Features of `b` at a single time `t`. Allocates; the hot paths use [`design_matrix`](@ref)
and one matrix-vector product instead.
"""
function eval_basis(b::AbstractBasis, t::Number)
    out = Vector{basis_eltype(b, typeof(t))}(undef, nbasis(b))
    return eval_basis!(out, b, t)
end

"""
    design_matrix(b::AbstractBasis, ts::AbstractVector) -> Matrix

`length(ts) × nbasis(b)` matrix whose row `i` is `eval_basis(b, ts[i])`, so that `B * β`
evaluates the basis expansion at every time at once.
"""
function design_matrix(b::AbstractBasis, ts::AbstractVector)
    M = Matrix{basis_eltype(b, eltype(ts))}(undef, length(ts), nbasis(b))
    row = Vector{eltype(M)}(undef, nbasis(b))
    @inbounds for (i, t) in enumerate(ts)
        eval_basis!(row, b, t)
        M[i, :] = row
    end
    return M
end

# ## Constant

"""
    ConstantBasis()

The intercept: a single column of ones.
"""
struct ConstantBasis <: AbstractBasis end

nbasis(::ConstantBasis) = 1
eval_basis!(out, ::ConstantBasis, t) = (out[1] = one(eltype(out)); out)

# ## Polynomial

"""
    PolynomialBasis(degree; center = 0, scale = 1)

Powers `u, u², …, u^degree` of `u = (t - center) / scale`. No intercept: add
[`ConstantBasis`](@ref) for one.

`center` and `scale` matter. On raw day indices `1:5114` an uncentred `degree = 2` gives
columns of magnitude `10³` and `10⁷` beside Fourier columns of magnitude 1, and the Hessian
is ill-conditioned enough to stall a quasi-Newton solver.
"""
struct PolynomialBasis{C <: Real} <: AbstractBasis
    degree::Int
    center::C
    scale::C

    # Inner, so that it cannot be bypassed by the constructor Julia generates by default.
    function PolynomialBasis(degree::Integer, center::C, scale::C) where {C <: Real}
        @argcheck degree >= 1 "a degree-0 polynomial is `ConstantBasis()`"
        @argcheck scale != 0
        return new{C}(Int(degree), center, scale)
    end
end

PolynomialBasis(degree::Integer; center = 0, scale = 1) =
    PolynomialBasis(degree, promote(center, scale)...)

nbasis(b::PolynomialBasis) = b.degree
paramtype(::PolynomialBasis{C}) where {C} = C

function eval_basis!(out, b::PolynomialBasis, t)
    u = (t - b.center) / b.scale
    p = one(eltype(out))
    @inbounds for k in 1:b.degree
        p *= u
        out[k] = p
    end
    return out
end

# ## Fourier

"""
    FourierBasis(period, order)

`[cos(ωt), sin(ωt), …, cos(order·ωt), sin(order·ωt)]` with `ω = 2π / period`. No intercept:
`ConstantBasis() + FourierBasis(T, d)` is the `2d+1` column parametrization
`[1, cos, sin, cos2, sin2, …]`, which is what [`TrigBasis`](@ref) spells.
"""
struct FourierBasis{P <: Real} <: AbstractBasis
    period::P
    order::Int

    # Inner, so that it cannot be bypassed by the constructor Julia generates by default.
    function FourierBasis(period::P, order::Integer) where {P <: Real}
        @argcheck period > 0
        @argcheck order >= 1 "a degree-0 Fourier basis is `ConstantBasis()`"
        return new{P}(period, Int(order))
    end
end

nbasis(b::FourierBasis) = 2 * b.order
paramtype(::FourierBasis{P}) where {P} = P

function eval_basis!(out, b::FourierBasis, t)
    # `2π * k * t / period`, associated exactly as `SmoothPeriodicStatsModels.trig_basis`
    # writes it, so the two design matrices agree bit for bit and coefficient vectors carry
    # over between the packages unchanged.
    @inbounds for k in 1:b.order
        s, c = sincos(2 * oftype(float(b.period), pi) * k * t / b.period)
        out[2k - 1] = c
        out[2k] = s
    end
    return out
end

"""
    TrigBasis(period, order) -> CombinedBasis

`ConstantBasis() + FourierBasis(period, order)`: the `2·order + 1` column trigonometric
polynomial `[1, cos(ωt), sin(ωt), …]`, column for column the design matrix that
`SmoothPeriodicStatsModels.trig_basis(period, order)` builds, so coefficient vectors
transfer between the two unchanged. `order = 0` degenerates to the intercept alone.
"""
TrigBasis(period::Real, order::Integer) =
    order == 0 ? ConstantBasis() : ConstantBasis() + FourierBasis(period, order)

# ## Indicator

"""
    IndicatorBasis(split_by, levels)

One-hot encoding of `split_by(t)` over `levels`, e.g. `IndicatorBasis(Dates.month, 1:12)`.

`levels` is required and ordered: it fixes what each column means, so a fitted basis keeps
its meaning when evaluated at a new time. Combining this with [`ConstantBasis`](@ref) gives
a rank-deficient design -- drop a level yourself if you want a reference category.
"""
struct IndicatorBasis{F, L <: AbstractVector} <: AbstractBasis
    split_by::F
    levels::L

    function IndicatorBasis(split_by::F, levels::L) where {F, L <: AbstractVector}
        @argcheck !isempty(levels)
        @argcheck allunique(levels)
        return new{F, L}(split_by, levels)
    end
end

nbasis(b::IndicatorBasis) = length(b.levels)

function eval_basis!(out, b::IndicatorBasis, t)
    g = b.split_by(t)
    @inbounds for (j, lvl) in enumerate(b.levels)
        out[j] = g == lvl
    end
    return out
end

# ## Sums

"""
    CombinedBasis(bases)

Concatenation of several bases, built with `+`. Columns appear in the order the bases do.
"""
struct CombinedBasis{T <: Tuple} <: AbstractBasis
    bases::T
end

Base.:+(a::AbstractBasis, b::AbstractBasis) = CombinedBasis((a, b))
Base.:+(a::CombinedBasis, b::AbstractBasis) = CombinedBasis((a.bases..., b))
Base.:+(a::AbstractBasis, b::CombinedBasis) = CombinedBasis((a, b.bases...))
Base.:+(a::CombinedBasis, b::CombinedBasis) = CombinedBasis((a.bases..., b.bases...))

nbasis(b::CombinedBasis) = sum(nbasis, b.bases)
paramtype(b::CombinedBasis) = mapreduce(paramtype, promote_type, b.bases; init = Bool)

function eval_basis!(out, b::CombinedBasis, t)
    i = 0
    for sub in b.bases
        n = nbasis(sub)
        eval_basis!(view(out, (i + 1):(i + n)), sub, t)
        i += n
    end
    return out
end

function Base.show(io::IO, b::CombinedBasis)
    join(io, b.bases, " + ")
    return
end
Base.show(io::IO, ::ConstantBasis) = print(io, "ConstantBasis()")
Base.show(io::IO, b::FourierBasis) = print(io, "FourierBasis(", b.period, ", ", b.order, ")")
Base.show(io::IO, b::PolynomialBasis) = print(io, "PolynomialBasis(", b.degree, ")")
Base.show(io::IO, b::IndicatorBasis) = print(io, "IndicatorBasis(", b.split_by, ", ", b.levels, ")")
