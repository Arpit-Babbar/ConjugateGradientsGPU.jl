using ConjugateGradients
using LinearAlgebra
using SparseArrays
using KernelAbstractions
using GPUArraysCore
using Test

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

function mymul!(x, A, y)
    x .= A*y
end

function test_cg_gpu(n=100; backend=CPU(), eltype = Float32)
    tA = KernelAbstractions.allocate(backend, eltype, n, n)
    b = KernelAbstractions.allocate(backend, eltype, n)
    true_x = KernelAbstractions.allocate(backend, eltype, n)
    tA_ = Array(sprandn(Float32, n, n, 0.1f0) + spdiagm(0=>fill(10.0f0, n)))
    b_ = rand(Float32, n)
    
    for j in 1:n
        for i in 1:n
            @allowscalar tA[i,j] = tA_[i,j]
        end
        @allowscalar b[j] = b_[j]
    end
    
    A = tA'*tA
    A_cpu = Array(A)
    b_cpu = Array(b)
    true_x_ = A_cpu\b_cpu
    
    for i in 1:n
        @allowscalar true_x[i] = true_x_[i]
    end

    x, exit_code, num_iters = cg((x,y) -> mymul!(x, A, y), b)
    @show x, exit_code, num_iters
    norm(true_x - x) < 1e-6
end

test_cg_gpu()

@testset "ConjugateGradients" begin
    @test test_cg_gpu()
end
