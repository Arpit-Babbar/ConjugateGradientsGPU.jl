include("test_base.jl")

@testset "ConjugateGradientsGPU" begin
    @test test_cg()
    @test test_bicgstab()
end
