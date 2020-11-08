using LinearAlgebra
using Test
using TridiagonalMatrixAlgorithm


N = 256

a = rand(N)
b = rand(N)
c = rand(N)
d = rand(N)

x = zeros(N)
tmp = similar(d)

M = Tridiagonal(a[2:N], b, c[1:N-1])


# ------------------------------------------------------------------------------
x = tridag(M, d)
@test isapprox(M * x, d)

tridag!(x, M, d, tmp)
@test isapprox(M * x, d)

@test (@allocated tridag!(x, M, d, tmp)) == 0


# ------------------------------------------------------------------------------
x = tridag(a, b, c, d)
@test isapprox(M * x, d)

tridag!(x, a, b, c, d, tmp)
@test isapprox(M * x, d)

@test (@allocated tridag!(x, a, b, c, d, tmp)) == 0
