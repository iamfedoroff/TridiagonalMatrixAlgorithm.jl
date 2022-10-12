module TridiagonalMatrixAlgorithm

import LinearAlgebra

export tridag!, tridag


# for GPU version of TDMA see
# https://gist.github.com/maleadt/1ec91b3b12ede9898958c95596cabe8b


"""
Tridiagonal matrix algorithm (aka Thomas algorithm, aka progonka).
https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

Solves a system of linear equations with tridiagonal matrix:

    a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i],   i=[1,N]

with a[1]=0 and c[N]=0.

Adapted from https://gist.github.com/maleadt/1ec91b3b12ede9898958c95596cabe8b
See also [Press et al., Numerical Recipes, 3rd edition (2007) (section 2.4)].
"""
function tridag!(
    x::AbstractVector{T},
    a::AbstractVector{T},
    b::AbstractVector{T},
    c::AbstractVector{T},
    d::AbstractVector{T},
    tmp::AbstractVector{T},
) where T
    N = length(x)

    beta = b[1]
    x[1] = d[1] / beta

    for j=2:N
        tmp[j] = c[j-1] / beta
        beta = b[j] - a[j] * tmp[j]
        if abs(beta) < 1e-12
            # This should only happen on last element of forward pass for
            # problems with zero eigenvalue. In that case the algorithmn is
            # still stable.
            break
        end
        x[j] = (d[j] - a[j] * x[j-1]) / beta
    end

    for j=1:N-1
        k = N - j
        x[k] = x[k] - tmp[k+1] * x[k+1]
    end

    return nothing
end


function tridag(
    a::AbstractVector{T},
    b::AbstractVector{T},
    c::AbstractVector{T},
    d::AbstractVector{T},
) where T
    x = similar(d)
    tmp = similar(d)
    tridag!(x, a, b, c, d, tmp)
    return x
end


# ******************************************************************************
function tridag!(
    x::AbstractVector{T},
    M::LinearAlgebra.Tridiagonal{T,<:Array},
    d::AbstractVector{T},
    tmp::AbstractVector{T},
) where T
    N = length(x)

    beta = M.d[1]
    x[1] = d[1] / beta

    for j=2:N
        tmp[j] = M.du[j-1] / beta
        beta = M.d[j] - M.dl[j-1] * tmp[j]
        if abs(beta) < 1e-12
            # This should only happen on last element of forward pass for
            # problems with zero eigenvalue. In that case the algorithmn is
            # still stable.
            break
        end
        x[j] = (d[j] - M.dl[j-1] * x[j-1]) / beta
    end

    for j=1:N-1
        k = N - j
        x[k] = x[k] - tmp[k+1] * x[k+1]
    end

    return nothing
end


function tridag(
    M::LinearAlgebra.Tridiagonal{T,<:Array},
    d::AbstractVector{T},
) where T
    x = similar(d)
    tmp = similar(d)
    tridag!(x, M, d, tmp)
    return x
end


end
