include("test_base.jl")

using Metal

@testset "ConjugateGradients" begin
    @test test_cg(backend = MetalBackend(), T = Float32)
    @test test_bicgstab(backend = MetalBackend(), T = Float32)
end
