# # Seasonal rain occurrence in Paris
#
# Daily rain / no rain at Paris, 1973-2025, as a Bernoulli whose success probability varies
# smoothly over the year:
#
#     p(t) = logistic( β₀ + Σₖ aₖ cos(2πkt/365.25) + bₖ sin(2πkt/365.25) )
#
# Run it with the test environment, which already has the solver stack:
#
#     julia --project=test examples/rain_occurrence.jl

using TimeVaryingParameters
using LinkFunctions: LogitLink
using Distributions: Bernoulli, succprob
using Dates
using OptimizationOptimJL, ForwardDiff   # together these install the default solver
using Printf

# ## Data
#
# Two columns, `time,rain`, so there is no need for a CSV reader here.

const DATA = joinpath(@__DIR__, "..", "docs", "src", "assets", "rain_paris.csv")

function load_rain(path)
    dates = Date[]
    rain = Bool[]
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue                       # header
        isempty(line) && continue
        d, r = split(line, ',')
        push!(dates, Date(d))
        push!(rain, parse(Bool, r))
    end
    return dates, rain
end

dates, rain = load_rain(DATA)
@printf("%d days, %s to %s, %.1f%% rainy\n",
    length(rain), first(dates), last(dates), 100 * count(rain) / length(rain))

# ## Model
#
# The period is the year. Day of year is the period index, so the problem is periodic and
# the design is built once at `1:366` rather than once per observation.

const T = 366
n2t = dayofyear.(dates)

spec = ParamSpec(p = TrigBasis(T, 2) => LogitLink())
fam = TimeVaryingFamily(Bernoulli, spec)

# ## Fit
#
# `init` gives a starting *value* for `p`, which goes onto the intercept through the link.

fitted = fit(fam, rain; n2t = n2t, nT = T, init = (p = 0.5,))

@printf("\ncoefficients: %s\n", round.(coef(fitted); digits = 4))
@printf("retcode:      %s\n", fitted.solution.retcode)

# ## Results
#
# `fitted(t)` is an ordinary `Distributions.Bernoulli`, so everything downstream is theirs.

curve_p = curve(fitted, :p, 1:T)
wettest = argmax(curve_p)
driest = argmin(curve_p)

@printf("\nfitted rain probability\n")
@printf("  wettest day %3d (%s): %.3f\n", wettest, Date(2001) + Day(wettest - 1), curve_p[wettest])
@printf("  driest  day %3d (%s): %.3f\n", driest, Date(2001) + Day(driest - 1), curve_p[driest])

println("\nmonthly fitted vs empirical")
for m in 1:12
    days = [t for t in 1:T if month(Date(2000) + Day(t - 1)) == m]   # 2000 is a leap year
    obs = rain[findall(in(Set(days)), n2t)]
    @printf("  %s  fitted %.3f   empirical %.3f\n",
        monthabbr(m), sum(curve_p[days]) / length(days), count(obs) / length(obs))
end

d = fitted(196)
println("\nfitted(196) = ", d, "  (a ", typeof(d), ", so succprob = ", round(succprob(d); digits = 4), ")")
