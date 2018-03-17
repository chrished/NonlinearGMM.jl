using NonlinearGMM
using Base.Test


# Seemingly Unrelated Regressions
# setup multiple equations with same covariates
const N = 1000
const nx = 2
const neq = 2
βh = rand((nx+1, neq))
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
bhat, V, Q, Ω = nlgmm(gsur, X, y, W, N, Nm, Nb; guess = vcat(zeros((nx+1)*neq), eye(neq)[:]), autodiff = :forward, maxit = 5000)

bgmm = bhat[1:(nx+1)*neq]
Σgmm = reshape(bhat[(nx+1)*neq+1:end], neq, neq)

# add Variance Covariance matrix GMM
# V is wrong need to divide by N! Where is it missing in my code?
bhatm1 = copy(bhat)
W[:,:] = inv(Ω)
for it = 1:10
    bhat, V, Q, Ω = nlgmm(gsur, X, y, W, N, Nm, Nb; guess = bhatm1, autodiff = :forward)
    if isapprox(bhat, bhatm1, atol=sqrt(eps()), rtol = sqrt(eps()))
        println("Converged iterated GMM")
        break
    end
    bhatm1[:] = bhat
    W[:,:] = inv(Ω)
end
