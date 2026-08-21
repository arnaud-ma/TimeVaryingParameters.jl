using TimeVaryingParameters
using LinkFunctions
using Distributions
using Dates
using GLMakie
# to trigger default optimizer
import ForwardDiff, OptimizationOptimJL

using CSV, DataFrames, Downloads
url = "https://archive-api.open-meteo.com/v1/archive?latitude=40.71&longitude=-74.00&start_date=2010-01-01&end_date=2023-12-31&daily=temperature_2m_mean,precipitation_sum&timezone=America%2FNew_York&format=csv"
df = CSV.read(Downloads.download(url), DataFrame, header = 4)
rename!(
    df, Dict(
        Symbol("time") => :date,
        Symbol("temperature_2m_mean (°C)") => :temp,
        Symbol("precipitation_sum (mm)") => :precip
    )
)

times = Dates.value.(df.date) .- Dates.value(first(df.date)) .+ 1


loss_rain(p, x) = -logpdf(Bernoulli(p[1]), x)


model_rain1 = TimeVaryingModel(
    bases = (p = ConstantBasis() + PolynomialBasis(1) + FourierBasis(365, 3),),
    links = (p = LinkFunctions.LogitLink(),),
)
fitted_rain1 = fit(
    model,
    loss_rain,
    df.precip .> 0,
    times;
    init = (p = 0.5,),
)

p_hat_rain1(t) = params.(fitted, t)[1]
lines(df.date, p_hat_rain1.(times))
ylims!(0, 1)


model_rain2 = TimeVaryingModel(
    bases = (p = PolynomialBasis(1) + IndicatorBasis(Dates.month, 1:12),),
    links = (p = LinkFunctions.LogitLink(),)
)

fitted_rain2 = fit(
    model_rain2,
    loss_rain,
    df.precip .> 0,
    times;
    init = (p = 0.5,),
)
p_hat2(t) = params.(fitted_rain2, t)[1]
lines!(df.date, p_hat2.(times))


loss_temp1(p, x) = -logpdf(Normal(p[1], p[2]), x)
model_temp1 = TimeVaryingModel(
    bases = (
        μ = ConstantBasis() + PolynomialBasis(1) + FourierBasis(365, 3),
        σ = ConstantBasis() + PolynomialBasis(1) + FourierBasis(365, 3),
    ),
    links = (
        μ = LinkFunctions.IdentityLink(),
        σ = LinkFunctions.LogLink(),
    )
)
fitted_temp1 = fit(
    model_temp1,
    loss_temp1,
    df.temp,
    times;
    init = (μ = mean(df.temp), σ = std(df.temp)),
)

μ_hat(t) = params.(fitted_temp1, t)[1]
σ_hat(t) = params.(fitted_temp1, t)[2]
lines(df.date, μ_hat.(times))
lines!(df.date, df.temp, color = (:red, 0.1))

temp_detrend = (df.temp .- μ_hat.(times)) ./ σ_hat.(times)
lines(df.date, temp_detrend)
hlines!([1.86, -1.86], color = :red)
