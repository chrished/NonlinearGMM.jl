module NonlinearGMM
    using Optim
    using ForwardDiff
    """
    Nonlinear GMM
    """
    function nlgmm(gfun, xdata, ydata, W, N, Nm, Nb; guess = ones(Nb), autodiff = :forward)
        # parameter estimation
        opt = nlgmm_opt(gfun, xdata, ydata, W, N, Nm, Nb; guess = guess, autodiff = autodiff)
        bgmm = opt.minimizer
        # calculcate variance covariance matrix
        V = nlgmm_covar(bgmm, gfun, xdata, zdata, ydata, W, N, Nm, Nb)
        return bgmm, V
    end

    """
    Parameter estimation
    """
    function nlgmm_opt(gfun, xdata, ydata, W, N, Nm, Nb; guess = ones(Nb), autodiff = :forward, maxit = 1000)
        f(x) = J(x, gfun, xdata, ydata, W, N, Nm, Nb)
        if autodiff == :forward
            obj = OnceDifferentiable(f, guess; autodiff = :forward)
            res = optimize(obj, guess, method = LBFGS(), show_trace = true, iterations = maxit, show_every = 10)

        else
            println("differentiation method not implemented")
        end
        return res
    end


    function J(b, gfun, xdata, ydata, W, N, Nm, Nb)
        gvec = zeros(eltype(b),(N,Nm))
        for i = 1:N
            gvec[i,:] = gfun(b, xdata[i,:], ydata[i,:])
        end
        gbar = mean(gvec,1)
        return (N*gbar * W * gbar')[1]
    end

    """
    Covariance Matrix given data, parameter estimates and weights
    """
    function nlgmm_covar(bhat, gfun, xdata, ydata, W, N, Nm, Nb)
        gvec = zeros(eltype(bhat),(N,Nm))
        for i = 1:N
            gvec[i,:] = gfun(bhat, xdata[i,:], ydata[i,:])
        end
        gbar = mean(gvec,1)[:]
        # construc Ω
        Ω = zeros(eltype(bhat),(Nm,Nm))
        for i = 1:N
            Ω += 1/N * (gvec[i,:]-gbar) *(gvec[i,:]-gbar)'
        end
        # contstruct Q
        Q = zeros(eltype(bhat),(Nm, Nb))
        for i = 1:N
            Q += 1/N * ForwardDiff.jacobian(par -> gfun(par, xdata[i,:], ydata[i,:]), bhat)' # check transpose
        end
        # check division by N
        return inv(Q'*W*Q)*(Q'*W*Ω*W*Q)*inv(Q'*W*Q)
    end

end # module
