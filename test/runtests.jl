using ConjugateGradients
using LinearAlgebra
using SparseArrays
using KernelAbstractions
using GPUArraysCore
using Test

# TODO - Bring these tests back?
# function test_cg(n=100)
#     tA = sprandn(n, n, 0.1) + spdiagm(0=>fill(10.0, n))
#     A = tA'*tA
#     b = rand(n)
#     true_x = A\b
#     x, exit_code, num_iters = cg((x,y) -> mul!(x, A, y), b)
#     norm(true_x - x) < 1e-6
# end

# function test_bicgstab(n=100)
#     A = sprandn(n, n, 0.1) + spdiagm(0=>fill(10.0, n))
#     b = rand(n)
#     true_x = A\b
#     x, exit_code, num_iters = bicgstab((x,y) -> mul!(x, A, y), b)
#     norm(true_x - x) < 1e-6
# end

# using Metal

function mymul!(x, A, y)
    x .= A*y
end

function test_cg_gpu(n=100; backend=CPU(), eltype = Float32)
    # tA_ = KernelAbstractions.allocate(backend, eltype, n, n)
    # tA = MtlArray(sprandn(Float32, n, n, 0.1f0)) + MtlArray(spdiagm(0=>fill(10.0f0, n)))
    
    tA = KernelAbstractions.allocate(backend, eltype, n, n)
    b = KernelAbstractions.allocate(backend, eltype, n)
    true_x = KernelAbstractions.allocate(backend, eltype, n)
    tA_cpu = Array(sprandn(Float32, n, n, 0.1f0) + spdiagm(0=>fill(10.0f0, n)))
    b_cpu = rand(Float32, n)
    @allowscalar for i in eachindex(tA_cpu)
        tA[i] = tA_cpu[i]
    end
    
    A = tA'*tA
    A_cpu = Array(A)
    b_cpu = Array(b)
    true_x_cpu = A_cpu\b_cpu
    
    @allowscalar for i in eachindex(b_cpu)
        b[i] = b_cpu[i]
        true_x[i] = true_x_cpu[i]
    end
    
    x, exit_code, num_iters = cg((x,y) -> mymul!(x, A, y), b)
    
    norm(true_x - x) < 1e-6
end

test_cg_gpu()

@testset "ConjugateGradients" begin
    @test test_cg_gpu()
end
