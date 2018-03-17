using NonlinearGMM
using Base.Test


# Seemingly Unrelated Regressions
# setup multiple equations with same covariates
const N = 500
const nx = 2
const neq = 2
βh = rand((nx+1, neq))*10
X = hcat(ones(N), rand((N, nx)))
y = X*βh .+ randn((N, neq))
Xbar = kron(eye(neq), X)
β = reshape(βh, (nx+1)*neq, 1)
ystack = reshape(y, N*neq, 1)
# standard SUR - iterated
bols = (X'X)\X'y
bols_stack = (Xbar'Xbar)\(Xbar'*ystack)

# Σ estimate
uhat = y .- X*bols
Σhat = zeros((neq, neq))
for i = 1:N
    Σhat += 1/N * uhat[i, :] * uhat[i,:]'
end
# now redo
# calculate V based on OLS, Hansen stacks the equations
# define Xbar as he does and stack β
# eq (10.17) in Bruce Hansens Econometrics book
Xi = zeros((neq,(nx+1)*neq))
Σhatinv = inv(Σhat)
XΣinvX = zeros(((nx+1)*neq, (nx+1)*neq))
XΣinvy = zeros(((nx+1)*neq, 1))

for i = 1:N
    Xi[1,1:nx+1] = X[i,:]
    Xi[2,nx+2:end] = X[i,:]
    XΣinvX +=  1/N * Xi' * Σhatinv * Xi
    XΣinvy += 1/N * Xi' * Σhatinv * y[i,:]
end
bsur = XΣinvX\XΣinvy
Vbsur = inv(N*XΣinvX) # should the N b here?

# GMM variant
# define moment function for observation i
# super slow!
function gsur(par, xdata, ydata)
    b = par[1:(nx+1)*neq]
    Σ = reshape(par[(nx+1)*neq+1:end], neq, neq)
    Σinv = inv(Σ)
    # first stack the parameters of the linear model and then Σ
    Xi = zeros((neq,(nx+1)*neq))
    Xi[1,1:nx+1] = xdata
    Xi[2,nx+2:end] = xdata
    u = (ydata-Xi*b)
    g1 = Xi'*Σinv * u
    g2 = (Σ - u*u')[:]
    return vcat(g1, g2)
end

btest = vcat(bsur[:], Σhat[:])
Nm = length(btest)
Nb = length(btest)
W = eye(length(btest))

@test isapprox(0.,NonlinearGMM.J(btest, gsur, X, y, W, N, Nm, Nb), atol=sqrt(eps()))

# now estimate gmm
bguess =vcat(zeros(bsur[:]), eye(neq)[:])

res = NonlinearGMM.nlgmm_opt(gsur, X, y, W, N, Nm, Nb; guess = bguess, autodiff = :forward, maxit = 5000)

bgmm = res.minimizer[1:(nx+1)*neq]
Σgmm = reshape(res.minimizer[(nx+1)*neq+1:end], neq, neq)
# add Variance Covariance matrix GMM
V = NonlinearGMM.nlgmm_covar(res.minimizer, gsur, X, y, W, N, Nm, Nb)
