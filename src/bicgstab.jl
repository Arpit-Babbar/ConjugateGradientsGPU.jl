using KernelAbstractions

# Data container
struct BiCGStabData{T <: Real, AbstractBackend <: KernelAbstractions.Backend}
    v::AbstractVector{T}
    t::AbstractVector{T}
    p::AbstractVector{T}
    phat::AbstractVector{T}
    s::AbstractVector{T}
    shat::AbstractVector{T}
    r::AbstractVector{T}
    rtld::AbstractVector{T}
    BiCGStabData(n::Int, T::Type, backend::KernelAbstractions.Backend) = new{T, typeof(backend)}(
        KernelAbstractions.zeros(backend, eltype(T), n),
        KernelAbstractions.zeros(backend, eltype(T), n),
        KernelAbstractions.zeros(backend, eltype(T), n),
        KernelAbstractions.zeros(backend, eltype(T), n),
        KernelAbstractions.zeros(backend, eltype(T), n),
        KernelAbstractions.zeros(backend, eltype(T), n),
        KernelAbstractions.zeros(backend, eltype(T), n),
        KernelAbstractions.zeros(backend, eltype(T), n))
end

# Main solver
function bicgstab!(A, b::AbstractVector{T}, x::AbstractVector{T};
                   tol::T=map(T,1e-6), maxIter::Int=100,
                   tolRho::T=map(T, 1e-40), precon=copy!,
                   data=BiCGStabData(length(b), T, get_backend(b))) where {T<:Real}
    bnrm2 = genblas_nrm2(b)
    if bnrm2 == zero(T)
        x .= zero(T)
        return 1, 0
    end
    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy!(one(T), b, data.r)
    residual = genblas_nrm2(data.r)/bnrm2
    if residual == zero(T)
        return 2, 0
    end
    alpha = one(T)
    omega = one(T)
    data.rtld .= data.r
    rho1 = zero(T)
    data.p .= data.r
    for iter = 1:maxIter
        rho = genblas_dot(data.rtld, data.r)
        if abs(rho) <= tolRho
            return -11, iter
        end
        if iter > 1
            beta  = (rho/rho1)*(alpha/omega)
            # p = r + beta*(p - omega*v)
            genblas_scal!(beta, data.p)
            genblas_axpy!(-(beta*omega), data.v, data.p)
            genblas_axpy!(one(T), data.r, data.p)
        end
        precon(data.phat, data.p)
        A(data.v, data.phat)
        alpha = rho/genblas_dot(data.rtld, data.v)
        # s = r - alpha*v
        data.s .= data.r
        genblas_axpy!(-alpha, data.v, data.s)
        # x = x + alpha*phat
        genblas_axpy!(alpha, data.phat, x)
        residual = genblas_nrm2(data.s)/bnrm2
        if residual <= tol
            return 31, iter
        end
        precon(data.shat, data.s)
        A(data.t, data.shat)
        omega = genblas_dot(data.t, data.s)/genblas_dot(data.t, data.t)
        # x = x + omega*shat
        genblas_axpy!(omega, data.shat, x)
        # r = s .- omega.*t
        data.r .= data.s
        genblas_axpy!(-omega, data.t, data.r)
        residual = genblas_nrm2(data.r)/bnrm2
        if residual <= tol
            return 32, iter
        end
        if abs(omega) < map(T, 1e-16)
            return -12, iter
        end
        rho1 = rho
    end
    return -2, maxIter
end

# API
function bicgstab(A, b::AbstractVector{T};
                  tol::T=map(T, 1e-6), maxIter::Int64=100,
                  tolRho::T=map(T, 1e-40), precon=copy!,
                  data=BiCGStabData(length(b), T, get_backend(b))) where {T<:Real}
    x = zeros(eltype(b), length(b))
    exit_code, num_iters = bicgstab!(A, b, x, tol=tol, maxIter=maxIter, tolRho=tolRho, precon=precon, data=data)
    return x, exit_code, num_iters
end

export BiCGStabData, bicgstab!, bicgstab
