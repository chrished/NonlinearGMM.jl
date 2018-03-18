using NonlinearGMM
using Base.Test
# test nonlinear gmm with OLS
N = 1000
nx = 3
X = hcat(ones(N), rand((N, nx)))
β = rand(nx+1)
y = X*β .+ randn(N)

bols = (X'*X)\(X'*y)

# moment function
g(β, xi, yi) = xi.*(yi - xi'*β)
# moment at bols
gvec = zeros((N,nx+1))
for i = 1:N
    gvec[i, :] = g(bols, X[i, :], y[i])
end

gbar = mean(gvec,1)
# is g appropriately defined?
@test isapprox(gbar, zeros(gbar), atol = sqrt(eps()))

# now estimate with GMM function and see if we get the same.
W = eye(nx+1)
Nm = nx+1
Nb = nx+1

opt = NonlinearGMM.nlgmm_opt(g, X, y, W, N, Nm, Nb; guess = ones(Nb), autodiff = :forward)

@test isapprox(opt.minimizer, bols, atol = sqrt(eps()))

bhat, V, Q, Ω = nlgmm(g, X, y, W, N, Nm, Nb; guess = zeros(bols), autodiff = :forward, maxit = 5000)
