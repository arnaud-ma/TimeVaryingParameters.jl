using TimeVaryingParameters
using Documenter

DocMeta.setdocmeta!(TimeVaryingParameters, :DocTestSetup, :(using TimeVaryingParameters); recursive=true)

makedocs(;
    modules=[TimeVaryingParameters],
    authors="arnaud-ma <arnaudma.code@gmail.com> and contributors",
    sitename="TimeVaryingParameters.jl",
    format=Documenter.HTML(;
        canonical="https://arnaud-ma.github.io/TimeVaryingParameters.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/arnaud-ma/TimeVaryingParameters.jl",
    devbranch="main",
)
