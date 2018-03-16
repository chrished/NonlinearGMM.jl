# Seemingly Unrelated Regressions
# setup multiple equations with same covariates
N = 10000
nx = 1
neq = 2
βh = rand((nx+1, neq))*10
X = rand((N, nx+1))
y = X*βb .+ randn((N, neq))
Xbar = kron(eye(neq), X)
β = reshape(βh, (nx+1)*neq, 1)
ystack = reshape(y, N*neq, 1)
# standard SUR - iterated
bols = (X'X)\X'y
bols_stack = (Xbar'Xbar)\(Xbar'*ystack)

# Σ estimate
uhat = y .- X*bols
Σhat = zeros((nx+1, nx+1))
for i = 1:N
    Σhat += 1/N * uhat[i, :] * uhat[i,:]'
end
# calculate V based on OLS, Hansen stacks the equations
# define Xbar as he does and stack β
# eq (10.17) in Bruce Hansens Econometrics book
# Σhatinv = inv(Σhat)
# XΣinv = zeros((nx+1, nx+1))
# for i = 1:N
#     XΣinv +=  1/N * X[i, :]
# end
# GMM variant
