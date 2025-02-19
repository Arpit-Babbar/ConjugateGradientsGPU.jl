include("test_base.jl")

@testset "ConjugateGradients" begin
    @test test_cg()
    @test test_bicgstab()
end
