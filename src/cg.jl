using KernelAbstractions

# Data container
struct CGData{ArrayType, AbstractBackend}
    r::ArrayType
    z::ArrayType
    p::ArrayType
    Ap::ArrayType
    backend::AbstractBackend
    CGData(n::Int, T::Type, backend) = new{typeof(KernelAbstractions.zeros(backend, eltype(T), n)), typeof(backend)}(
        KernelAbstractions.zeros(backend, eltype(T), n), KernelAbstractions.zeros(backend, eltype(T), n),
        KernelAbstractions.zeros(backend, eltype(T), n), KernelAbstractions.zeros(backend, eltype(T), n))
end

# Solves for x
function cg!(A, b::AbstractVector{T}, x::AbstractVector{T};
             tol::Float64=1e-6, maxIter::Int64=100,
             precon=copy!,
             data=CGData(length(b), T, get_backend(b))) where {T<:Real}
    if genblas_nrm2(b) == 0.0f0
        x .= 0.0f0
        return 1, 0
    end
    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
    residual_0 = genblas_nrm2(data.r)
    if residual_0 <= tol
        return 2, 0
    end
    precon(data.z, data.r)
    data.p .= data.z
    for iter = 1:maxIter
        A(data.Ap, data.p)
        gamma = genblas_dot(data.r, data.z)
        alpha = gamma/genblas_dot(data.p, data.Ap)
        if alpha == Inf || alpha < 0
            return -13, iter
        end
        # x += alpha*p
        genblas_axpy!(alpha, data.p, x)
        # r -= alpha*Ap
        genblas_axpy!(-alpha, data.Ap, data.r)
        residual = genblas_nrm2(data.r)/residual_0
        if residual <= tol
            return 30, iter
        end
        precon(data.z, data.r)
        beta = genblas_dot(data.z, data.r)/gamma
        # p = z + beta*p
        genblas_scal!(beta, data.p)
        genblas_axpy!(1.0f0, data.z, data.p)
    end
    return -2, maxIter
end

# API
function cg(A, b::AbstractVector{T};
            tol::Float64=1e-6, maxIter::Int64=100,
            precon=copy!,
            data=CGData(length(b), T, get_backend(b))) where {T<:Real}
    backend = get_backend(b)
    x = KernelAbstractions.zeros(backend, eltype(b), length(b))
    exit_code, num_iters = cg!(A, b, x, tol=tol, maxIter=maxIter, precon=precon, data=data)
    return x, exit_code, num_iters
end

export CGData, cg!, cg
