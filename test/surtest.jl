using NonlinearGMM
using Base.Test


# Seemingly Unrelated Regressions
# setup multiple equations with same covariates
N = 1000
nx = 2
neq = 2

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
function unpack_par(par)
    b = par[1:(nx+1)*neq]
    Σ = diagm(par[(nx+1)*neq+1:(nx+1)*neq+neq])
    stop = (nx+1)*neq+neq
    start = (nx+1)*neq+neq
    for ieq = 2:neq
        start = stop + 1
        stop = start + (neq -ieq)
        Σ += diagm(par[start:stop] , (ieq-1)) + diagm(par[start:stop] , -(ieq-1))
    end
    return b, Σ
end
function gsur(par, xdata, ydata)
    b, Σ = unpack_par(par)
    #Σ = reshape(par[(nx+1)*neq+1:end], neq, neq)
    Σinv = inv(Σ)
    # first stack the parameters of the linear model and then Σ
    Xi = zeros((neq,(nx+1)*neq))
    Xi[1,1:nx+1] = xdata
    Xi[2,nx+2:end] = xdata
    u = (ydata-Xi*b)
    g1 = Xi'*Σinv * u
    G2 = (Σ - u*u')
    g2 = zeros(eltype(par),Int(neq^2 - (neq*(neq-1))/2))
    start = 1
    stop = 1
    for d = 1:neq
        stop = start + neq - d
        g2[start:stop] = diag(G2, d-1)
        start = stop +1
    end

    return vcat(g1, g2)
end

btest = vcat(bsur[:], diag(Σhat),  diag(Σhat, 1))


Nm = length(btest)
Nb = length(btest)
W = eye(length(btest))

@test isapprox(0.,NonlinearGMM.J(btest, gsur, X, y, W, N, Nm, Nb), atol=sqrt(eps()))

# now estimate gmm
bhat, V, Q, Ω = nlgmm(gsur, X, y, W, N, Nm, Nb; guess = vcat(ones(bsur), ones(neq), zeros(Int(neq*(neq-1)/2))), autodiff = :forward, maxit = 5000)

bgmm, Σgmm = unpack_par(bhat)

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
bgmm_it = bhat[1:(nx+1)*neq]

@test isapprox(bgmm_it, bols_stack, atol=sqrt(eps()))
