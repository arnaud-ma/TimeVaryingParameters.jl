using TimeVaryingParameters
using Test
using Aqua
using JET

@testset "TimeVaryingParameters.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(TimeVaryingParameters)
    end
    # @testset "Code linting (JET.jl)" begin
    #     JET.test_package(TimeVaryingParameters; target_defined_modules = true)
    # end
    # Write your tests here.
end
