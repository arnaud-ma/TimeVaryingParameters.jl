# TimeVaryingParameters [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://arnaud-ma.github.io/TimeVaryingParameters.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://arnaud-ma.github.io/TimeVaryingParameters.jl/dev/) [![Build Status](https://github.com/arnaud-ma/TimeVaryingParameters.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/arnaud-ma/TimeVaryingParameters.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/arnaud-ma/TimeVaryingParameters.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/arnaud-ma/TimeVaryingParameters.jl) [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Statistical models whose parameters vary smoothly with time, built out of distributions that do not.

Each parameter is a basis expansion pushed through a link,

```
θⱼ(t) = linkⱼ⁻¹( Bⱼ(t) · βⱼ )
```

so a model is one basis and one link per named parameter. Paired with a distribution constructor, evaluating it at a time gives an ordinary
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl) object — `logpdf`, `rand`,
`quantile` and the rest come from there rather than being reimplemented here.

## Example

A Normal whose mean and standard deviation both follow the seasons, with `σ(t) > 0` enforced
by the link:

```julia
using TimeVaryingParameters, LinkFunctions, Distributions, Dates
using OptimizationOptimJL, ForwardDiff   # the solver stack

T = 366   # period, in days

fam = TimeVaryingFamily(Normal, ParamSpec(
    μ = TrigBasis(T, 2) => IdentityLink(),
    σ = TrigBasis(T, 2) => LogLink(),
))

fitted = fit(fam, y; n2t = dayofyear.(dates), nT = T, init = (μ = 0.0, σ = 1.0))

coef(fitted, :σ)            # that parameter's coefficient block
curve(fitted, :σ, 1:T)      # σ(t) over one period
fitted(196)                 # a Normal, from Distributions.jl
```

`examples/rain_occurrence.jl` is a complete runnable version on 53 years of Paris rainfall.

## Bases

| Basis | Columns | Notes |
|---|---|---|
| `ConstantBasis()` | 1 | the intercept |
| `FourierBasis(period, order)` | `2·order` | `[cos ωt, sin ωt, …]`, no intercept |
| `TrigBasis(period, order)` | `2·order + 1` | `ConstantBasis() + FourierBasis(…)` |
| `PolynomialBasis(d; center, scale)` | `d` | centre and scale it, or the Hessian suffers |
| `IndicatorBasis(f, levels)` | `length(levels)` | one-hot, e.g. `(Dates.month, 1:12)` |
| `BSplineBasis(lower, upper, n; degree = 3)` | `n + degree - 1` | clamped B-splines on `n` equally spaced knots; also `BSplineBasis(knots)` |
| `CyclicBSplineBasis(period, n; degree = 3)` | `n` | periodic B-splines on `n` equally spaced knots |

Both B-spline bases sum to one at every point, so they contain the intercept: do not add a
`ConstantBasis()` to them.

They compose with `+`, so `ConstantBasis() + PolynomialBasis(1) + FourierBasis(365, 3)` is a
trend plus a seasonal cycle.

## Links

Links come from [LinkFunctions.jl](https://github.com/giovannitinervia9/LinkFunctions.jl) —
`IdentityLink`, `LogLink`, `LogitLink`, `BoundedLink(a, b)` and friends — and any callable
mapping the linear predictor onto the parameter domain works too.

This package adds one. `IntervalLink(a, b)` is the same map as `BoundedLink(a, b)`,
`θ = a + (b - a)·logistic(η)`, written so that it saturates rather than overflows:
`BoundedLink` evaluates `(a + b·exp(η))/(1 + exp(η))`, which is `Inf/Inf` — `NaN`, and a `NaN`
gradient with it — once `η > 709`, somewhere a line search can step. Prefer it wherever a
parameter is confined to an interval. `IntervalLink(-1, 1)` is the stationarity constraint of
an AR(1) coefficient.

## Weights, and the periodic fast path

`fit` takes `weights`, one per observation, which is what lets this serve as the M-step of an
EM algorithm.

When the problem is periodic, pass `n2t` (each observation's period index) and `nT`. The
design is then built once at `1:nT` and the likelihood accumulated per period index rather
than per observation. On 50 000 daily observations with `T = 366` and two seasonal
parameters that is **33× faster** per objective evaluation and **13× faster** per gradient
than the generic path, and allocates 6 KB per call instead of 16 MB.

For objectives that are not a loglikelihood, `fit(spec, loss, y, ts; init)` minimises
`Σₙ wₙ · loss(θ(tsₙ), yₙ)` for any `loss(params::Tuple, y)`.

## A note on performance

Pass the distribution type directly — `TimeVaryingFamily(Normal, spec)`. It gets wrapped in a
singleton carrying the type as a *type parameter*, because storing a bare `UnionAll` in a
field makes the objective infer as `Any` and run an order of magnitude slower. Reach for a
closure only when you need one, for instance to skip argument checking:

```julia
TimeVaryingFamily((μ, σ) -> Normal(μ, σ; check_args = false), spec)
```

## Compatibility with SmoothPeriodicStatsModels.jl

`design_matrix(TrigBasis(T, d), 1:T)` is **bit-identical** to
[`SmoothPeriodicStatsModels.trig_basis(T, d)`](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl),
and `IdentityLink`, `LogLink`, `LogitLink`, `IntervalLink(-1, 1)` reproduce that package's
`μₜ`, `σₜ`, `αₜ`, `ρₜ`. Coefficient vectors therefore carry over between the two unchanged.
Both properties are asserted in the test suite.
