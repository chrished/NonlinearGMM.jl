module NonlinearGMM
    using Optim
    """
    Nonlinear GMM
    """
    function nlgmm(gfun, xdata, ydata, W, N, Nm, Nb; guess = ones(Nb), autodiff = :forward)
        # parameter estimation
        opt = nlgmm_opt(gfun, xdata, ydata, W, N, Nm, Nb; guess = guess, autodiff = autodiff)
        bgmm = opt.minimizer
        # calculcate variance covariance matrix

    end

    """
    Parameter estimation
    """
    function nlgmm_opt(gfun, xdata, ydata, W, N, Nm, Nb; guess = ones(Nb), autodiff = :forward)
        if autodiff == :forward
            f(x) = J(x, gfun, xdata, ydata, W, N, Nm, Nb)
            obj = TwiceDifferentiable(f, guess; autodiff = :forward)
        end
        res = optimize(obj, guess, Newton())
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
    function nlgmm_covar(bhat, gfun, xdata, zdata, ydata, W, N, Nm, Nb)
        # construc Ω
        # contstruct Q
    end

end # module
