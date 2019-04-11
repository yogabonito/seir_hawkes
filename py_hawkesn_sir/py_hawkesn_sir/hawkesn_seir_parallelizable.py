import numbers

from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import matplotlib.pyplot as plt

llf_factor = 10**6  # factor for avoiding a too flat likelihood function

def exp_intensity(beta, sigma, gamma, n, history, sum_less_equal=True):
    """
    Calculate the (exponential) intensity of a HawkesN process.

    Parameters
    ----------
    beta : float
        Parameter beta of the corresponding SEIR model.
    sigma : float or None
        Parameter sigma of the corresponding SEIR model. If None,
        `sigma`==`gamma` is assumed.
    gamma : float
        Parameter gamma of the corresponding SEIR model.
    n : float or int
        The size of the population.
    history : np.array
        Array containing the process' jump times.
    sum_less_equal : bool, default: True
        If True, we sum over all event times <= time t. Otherwise, we sum
        over all event times < time t.


    Returns
    -------
    exp_intensity_ : function
        A function of the time (expecting a float or np.array as argument).

    """
    def exp_intensity_(t):
        """
        Parameters
        ----------
        t : float or np.array
            If `t` is a float, then it represents a point in time. If it is
            a 1-dimensional array, then it represents an array of points in
            time.

        Returns
        -------
        result : float or np.array
            If `t` is a float, then the intensity of the HawkesN process at
            time `t` is returned. If `t` is a 1-dimensional array, then an
            array is returned. Each entry of it represents the intensity of
            the HawkesN process at the time that was specified by the
            corresponding entry in `t`.
        """
        nonlocal sigma
        if sigma is None:
            sigma = gamma
        #if np.isnan([beta, sigma, gamma, n]).any():
        #    raise RuntimeError("One of the arguments to exp_intensity is nan: "
        #                       "beta" + str(beta) + ", sigma" + str(sigma) +
        #                       ", gamma" + str(gamma) + ", n" + str(n) +
        #                       ", history" + str(history))
        if isinstance(t, numbers.Number):
            t = np.array([t])
        result = np.empty(t.shape)
        for index, time in enumerate(t):
            if sum_less_equal:
                history_until_t = history[history <= time]
            else:
                history_until_t = history[history < time]

            if sigma != gamma:
                result[index] = np.sum(
                    np.exp(-sigma * (time - history_until_t))
                    -
                    np.exp(-gamma * (time - history_until_t))
                )
            else:
                result[index] = np.sum(
                    (time - history_until_t) *
                    np.exp(-gamma * (time - history_until_t))
                )

            result[index] *= (1 - np.count_nonzero(history <= time)/n)
            # print("intensity at index", index, "is", result[index]*scale*decay)
        if sigma != gamma:
            result *= beta * sigma / (gamma - sigma)
        else:
            result *= beta * gamma
        return result
    return exp_intensity_


def plot_exp_intensity(t_max, intensity=None, beta=None, sigma=None,
                       gamma=None, n=None, history=None, width=5.51, height=4,
                       n_xticks=6, step=.01, fname=None, **kwargs):
    """
    Plot (or save the plot of) the exponential intensity function from t=0
    until t=t_max.

    Parameters
    ----------
    t_max : float
        Define the time horizon of the plot. The time axis will contain
        values from 0 to t_max.
    intensity : function or None
        If None, then the arguments scale, decay, and n must be provided in
        order to compute an intensity function. If exp_intensity is already
        a function (taking the time as only argument) then scale, decay, n,
         and history are ignored.
    beta : float
        See corresponding argument in :func:`exp_intensity`. Ignored if
        exp_intensity is provided as argument.
    sigma : float
        See corresponding argument in :func:`exp_intensity`. Ignored if
        exp_intensity is provided as argument.
    gamma : float
        See corresponding argument in :func:`exp_intensity`. Ignored if
        exp_intensity is provided as argument.
    n : int or None
        Population size. Ignored if exp_intensity is provided as argument.
    history : np.array or None
        One dimensional array containing the event times. The event times
        must be sorted in ascending order. Ignored if exp_intensity is
        provided as argument.
    width : float, default: 5.51
        Width of the plot.
    height : float, default: 4
        Height of the plot.
    n_xticks : int (must be non-negative)
        Number of ticks on the time axis.
    step : float
        Step size for drawing the function graph.
    fname : str or None
        Name (without extension) of the file the plot is saved to. If
        `None`, the plot is not saved.
    """
    if intensity is None:
        intensity = exp_intensity(beta, sigma, gamma, n, history, **kwargs)

    t = np.arange(0, t_max, step)

    plt.figure(dpi=300, figsize=(width, height))
    plt.plot(t, intensity(t))

    plt.xlabel("$t$")
    plt.xlim(0, t_max)
    plt.xticks(np.linspace(0, t_max, n_xticks))

    plt.ylabel("Intensity")

    plt.grid()

    title = "Intensity of a SEIR-related HawkesN process"
    if history is not None and beta is not None and sigma is not None \
            and sigma is not None and n is not None:
        title += " with\nevent history \{" \
                 + ",".join(str(i) for i in history[:4]) \
                 + (", ..." if len(history) > 4 else "") \
                 + "\} and parameters: $\\beta=" + str(beta) \
                 + "$, $\\sigma=" + str(sigma) + "$, $\\gamma=" + str(gamma) + \
                 "$, $N$=" + str(n)
    title += "."
    plt.title(title)

    if fname is not None:
        plt.savefig(fname + ".pdf")


def llf_sigma_neq_gamma(beta, sigma, gamma, n, history, sum_less_equal=True):
    """

    Parameters
    ----------
    beta : float
        See corresponding argument in :func:`exp_intensity`.
    sigma : float
        See corresponding argument in :func:`exp_intensity`.
    gamma : float
        See corresponding argument in :func:`exp_intensity`.
    n : float
        See corresponding argument in :func:`exp_intensity`.
    history : np.array
        See corresponding argument in :func:`exp_intensity`.
    sum_less_equal : bool, default: True
        See corresponding argument in :func:`exp_intensity`.


    Returns
    -------
    llf : numpy.float64
        The log-likelihood for the parameters passed as arguments.
    """
    intensity = exp_intensity(beta, sigma, gamma, n, history,
                              sum_less_equal=sum_less_equal)

    sum_log_arg = intensity(history[history > 0])
    if sum_log_arg[0] <=0:
        sum_log_arg = sum_log_arg[1:]
    sum_part = np.sum(np.log(sum_log_arg))

    int_part = 0
    for i in range(len(history) - 1):
        int_part += (n - (i + 1)) / n * np.sum(
            (
                np.exp(-sigma * (history[i] - history[:i + 1]))
                -
                np.exp(-sigma * (history[i + 1] - history[:i + 1]))
            ) / sigma
            -
            (
                np.exp(-gamma * (history[i] - history[:i + 1]))
                -
                np.exp(-gamma * (history[i + 1] - history[:i + 1]))
            ) / gamma
        )
    int_part *= (beta * sigma / (gamma-sigma))
    # print("sum:", sum_part)
    # print("integral:", int_part)
    # print("*** llf:", sum_part - int_part)
    return sum_part - int_part


def llf_sigma_eq_gamma(beta, gamma, n, history, sum_less_equal=True):
    """

    Parameters
    ----------
    beta : float
        See corresponding argument in :func:`exp_intensity`.
    gamma : float
        See corresponding argument in :func:`exp_intensity`.
    n : float
        See corresponding argument in :func:`exp_intensity`.
    history : np.array
        See corresponding argument in :func:`exp_intensity`.
    sum_less_equal : bool, default: True
        See corresponding argument in :func:`exp_intensity`.


    Returns
    -------
    llf : numpy.float64
        The log-likelihood for the parameters passed as arguments.
    """
    intensity = exp_intensity(beta, gamma, gamma, n, history,
                              sum_less_equal=sum_less_equal)
    
    sum_log_arg = intensity(history[history > 0])
    if sum_log_arg[0] <=0:
        sum_log_arg = sum_log_arg[1:]
    sum_part = np.sum(sum_log_arg)

    int_part = 0
    for i in range(len(history) - 1):
        int_part += (n - (i + 1)) / n * np.sum(
            np.exp(-gamma * (history[i] - history[:i + 1]))
            * (gamma * (history[i] - history[:i + 1]) + 1)
            -
            np.exp(-gamma * (history[i + 1] - history[:i + 1]))
            * (gamma * (history[i + 1] - history[:i + 1]) + 1)
        ) / gamma
    int_part *= beta
    # print("sum:", sum_part)
    # print("integral:", int_part)
    # print("*** llf:", sum_part - int_part)
    return sum_part - int_part


def llf(beta, sigma, gamma, n, history, sum_less_equal=True):
    if sigma != gamma and sigma is not None:
        return llf_sigma_neq_gamma(beta, sigma, gamma, n, history,
                                   sum_less_equal)
    else:
        return llf_sigma_eq_gamma(beta, gamma, n, history, sum_less_equal)


def fit_sigma_neq_gamma(history, beta_start=None, sigma_start=None,
                        gamma_start=None, n_start=None, estimate_n_only=False):
    """
    Parameters
    ----------
    history : np.array
        1-dimensional array containing the event times in ascending order.
    beta_start : float
        Starting value for the likelihood optimization.
    sigma_start : float
        Starting value for the likelihood optimization.
    gamma_start : float
        Starting value for the likelihood optimization.
    n_start : float or None, default: None
        Starting value for the likelihood optimization. If None, a value is
        chosen based on the number of events contained in the `history`.
    estimate_n_only : bool, default: False
        If True, `beta`, `sigma` and `gamma` are considered to be fixed and
        only :math:`N` is fitted. Otherwise, `beta`, `sigma` and `gamma` are
        fitted together with :math:`N`.

    References
    ----------
    This method uses the L-BFGS algorithm (see [1]_).
    .. [1] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    """

    if estimate_n_only and \
            (beta_start is None or sigma_start is None or gamma_start is None):
        raise Exception("If beta, sigma, and gamma are fixed, their values "
                        "must be provided!")

    if n_start is None:
        n_start = len(history) + .5

    def negative_llf(beta_sigma_gamma_n):
        """
        Parameters
        ----------
        beta_sigma_gamma_n : np.array (shape (4))
            Values for the parameters beta, sigma, gamma, and N in a single
            array.

        Returns
        -------
        neg_llf : float
            The negative log-likelihood.
        """
        beta, sigma, gamma, n = tuple(beta_sigma_gamma_n)
        if sigma == gamma:
            sigma += 1e-7
        return -llf_factor * llf(beta=beta, sigma=sigma, gamma=gamma, n=n,
                                 history=history, sum_less_equal=False)

    def negative_llf_separate_params(n, beta, sigma, gamma):
        """
        Same as :func:`negative_llf` but taking the parameters `n`, `beta`,
        `sigma`, and `gamma` as separate arguments. This makes the function
        suitable for likelihood maximization in only one parameter (`n`) with
        fixed values for `beta`, `sigma`, and `gamma`.
        """
        if sigma == gamma:
            sigma += 1e-7
        return -llf_factor * llf(beta=beta, sigma=sigma, gamma=gamma, n=n,
                                 history=history, sum_less_equal=False)

    def negative_llf_gradient(beta_sigma_gamma_n):
        beta, sigma, gamma, n = tuple(beta_sigma_gamma_n)
        if sigma == gamma:
            sigma += 1e-7
        return -llf_factor * llf_gradient(beta=beta, sigma=sigma, gamma=gamma,
                                          n=n, history=history,
                                          sum_less_equal=False)

    def negative_llf_gradient_separate_params(n, beta, sigma, gamma):
        if sigma == gamma:
            sigma += 1e-7
        return -llf_factor * dllf_dn_sigma_neq_gamma(beta=beta, sigma=sigma,
                                                     gamma=gamma, n=n,
                                                     history=history,
                                                     sum_less_equal=False)

    eps = np.finfo(float).eps

    if estimate_n_only:
        return fmin_l_bfgs_b(
            func=negative_llf_separate_params,  # minimize this
            x0=np.array([n_start]),  # initial guess
            args=(beta_start, sigma_start, gamma_start),  # additional args to func&fprime
            fprime=negative_llf_gradient_separate_params,
            bounds=[(len(history), None)],
            iprint=1
        )

    else:
        return fmin_l_bfgs_b(
            func=negative_llf,  # minimize this
            x0=np.array([beta_start,
                         sigma_start,
                         gamma_start,
                         n_start]),  # initial guess
            fprime=negative_llf_gradient,
            bounds=[(eps, None),
                    (eps, None),
                    (eps, None),
                    (len(history), None)],
            factr=10,
            iprint=1
        )


def fit_sigma_eq_gamma(history, beta_start=None, gamma_start=None,
                       n_start=None, estimate_n_only=False):
    """
    Parameters
    ----------
    history : np.array
        1-dimensional array containing the event times in ascending order.
    beta_start : float
        Starting value for the likelihood optimization.
    gamma_start : float
        Starting value for the likelihood optimization.
    n_start : float or None, default: None
        Starting value for the likelihood optimization. If None, a value is
        chosen based on the number of events contained in the `history`.
    estimate_n_only : bool, default: False
        If True, `beta` and `gamma` are considered to be fixed and only
        :math:`N` is fitted. Otherwise, `beta` and `gamma` are fitted together
        with :math:`N`.

    References
    ----------
    This method uses the L-BFGS algorithm (see [1]_).
    .. [1] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    """

    if estimate_n_only and (beta_start is None or gamma_start is None):
        raise Exception("If beta and gamma are fixed, their values "
                        "must be provided!")

    if n_start is None:
        n_start = len(history) + .5

    def negative_llf(beta_gamma_n):
        """
        Parameters
        ----------
        beta_gamma_n : np.array (shape (4))
            Values for the parameters beta, gamma, and N in a single array.

        Returns
        -------
        neg_llf : float
            The negative log-likelihood.
        """
        beta, gamma, n = tuple(beta_gamma_n)
        return -llf(beta=beta, sigma=None, gamma=gamma, n=n, history=history,
                    sum_less_equal=False)

    def negative_llf_separate_params(n, beta, gamma):
        """
        Same as :func:`negative_llf` but taking the parameters `n`, `beta`,
        and `gamma` as separate arguments. This makes the function suitable for
        likelihood maximization in only one parameter (`n`) with fixed values
        for `beta` and `gamma`.
        """
        return -llf(beta=beta, sigma=None, gamma=gamma, n=n, history=history,
                    sum_less_equal=False)

    def negative_llf_gradient(beta_gamma_n):
        beta, gamma, n = tuple(beta_gamma_n)
        return -llf_gradient(beta=beta, sigma=None, gamma=gamma, n=n,
                             history=history, sum_less_equal=False)

    def negative_llf_gradient_separate_params(n, beta, gamma):
        return -dllf_dn_sigma_eq_gamma(beta=beta, gamma=gamma, n=n,
                                       history=history, sum_less_equal=False)

    eps = np.finfo(float).eps

    if estimate_n_only:
        return fmin_l_bfgs_b(
            func=negative_llf_separate_params,  # minimize this
            x0=np.array([n_start]),  # initial guess
            args=(beta_start, gamma_start),  # additional args to func&fprime
            fprime=negative_llf_gradient_separate_params,
            bounds=[(len(history), None)],
            iprint=1
        )

    else:
        return fmin_l_bfgs_b(
            func=negative_llf,  # minimize this
            x0=np.array([beta_start, gamma_start, n_start]),  # initial guess
            fprime=negative_llf_gradient,
            bounds=[(eps, None), (eps, None), (len(history), None)],
            iprint=1
        )


def dllf_dbeta_sigma_neq_gamma(beta, sigma, gamma, n, history,
                               sum_less_equal=True):
    """
    Parameters
    ----------
    beta : float
        See corresponding argument in :func:`exp_intensity`.
    sigma : float
        See corresponding argument in :func:`exp_intensity`.
    gamma : float
        See corresponding argument in :func:`exp_intensity`.
    n : float
        See corresponding argument in :func:`exp_intensity`.
    history : np.array
        See corresponding argument in :func:`exp_intensity`.
    sum_less_equal : bool, default: True
        See corresponding argument in :func:`exp_intensity`.
        Note that this argument does not affect the derivative w.r.t. the beta
        parameter. Thus, the return value does not depend on it.


    Returns
    -------
    derivative_wrt_beta : float
        The derivative (w.r.t. the beta parameter) of the log-likelihood
        function given the `history` and evaluated at the parameters
        `beta`, `sigma`, `gammma`, and `n`.
    """
    sum_part = np.count_nonzero(history) / beta
    int_part = 0
    for l in range(len(history) - 1):
        int_part += (n - (l + 1)) / n * (
            np.sum(
                np.exp(-sigma * (history[l] - history[:l + 1]))
                -
                np.exp(-sigma * (history[l + 1] - history[:l + 1]))
            ) / sigma
            -
            np.sum(
                np.exp(-gamma * (history[l] - history[:l + 1]))
                -
                np.exp(-gamma * (history[l + 1] - history[:l + 1]))
            ) / gamma
        )

    return sum_part - int_part * sigma / (gamma - sigma)


def dllf_dbeta_sigma_eq_gamma(beta, gamma, n, history, sum_less_equal=True):
    """
    Parameters
    ----------
    beta : float
        See corresponding argument in :func:`exp_intensity`.
    gamma : float
        See corresponding argument in :func:`exp_intensity`.
    n : float
        See corresponding argument in :func:`exp_intensity`.
    history : np.array
        See corresponding argument in :func:`exp_intensity`.
    sum_less_equal : bool, default: True
        See corresponding argument in :func:`exp_intensity`.
        Note that this argument does not affect the derivative w.r.t. the beta
        parameter. Thus, the return value does not depend on it.


    Returns
    -------
    derivative_wrt_beta : float
        The derivative (w.r.t. the beta parameter) of the log-likelihood
        function given the `history` and evaluated at the parameters
        `beta`, `gammma`, and `n`.
    """
    sum_part = np.count_nonzero(history) / beta
    # print("#" * 80, "\n", "SCALE", sep="")
    # print("#" * 80)
    # print("addend sum:", addend_sum)
    # print("#" * 80)
    # print("#" * 80)
    int_part = 0
    for l in range(len(history) - 1):
        mg_tl_minus_tj = -gamma * (history[l] - history[:l + 1])
        mg_tlplus1_minus_tj = -gamma * (history[l + 1] - history[:l + 1])

        int_part += (n - (l + 1)) / n * np.sum(
                np.exp(mg_tlplus1_minus_tj) * (mg_tlplus1_minus_tj - 1)
                -
                np.exp(mg_tl_minus_tj) * (mg_tl_minus_tj - 1)
            )
    return sum_part - int_part / gamma


def dllf_dsigma_sigma_neq_gamma(beta, sigma, gamma, n, history,
                                sum_less_equal=True):
    """
    Parameters
    ----------
    beta : float
        See corresponding argument in :func:`exp_intensity`.
    sigma : float
        See corresponding argument in :func:`exp_intensity`.
    gamma : float
        See corresponding argument in :func:`exp_intensity`.
    n : float
        See corresponding argument in :func:`exp_intensity`.
    history : np.array
        See corresponding argument in :func:`exp_intensity`.
    sum_less_equal : bool, default: True
        See corresponding argument in :func:`exp_intensity`.

    Returns
    -------
    derivative_wrt_sigma : float
        The derivative (w.r.t. the sigma parameter) of the log-likelihood
        function given the `history` and evaluated at the parameters
        `beta`, `sigma`, `gammma`, and `n`.
    """
    his_after_zero = history[history > 0]
    gms = float(gamma - sigma)
    addend_sum = np.count_nonzero(history) * gamma / (sigma * gms)

    for index, event_time in enumerate(his_after_zero):
        if sum_less_equal:
            ti_minus_tj = event_time - history[history <= event_time]
        else:
            ti_minus_tj = event_time - history[history < event_time]

        exp_sigma = np.exp(-sigma * ti_minus_tj)
        exp_gamma = np.exp(-gamma * ti_minus_tj)

        numerator = np.sum(-ti_minus_tj * exp_sigma)
        # print("addend_sum_tmp numerator:", addend_sum_tmp)
        denominator = np.sum(exp_sigma - exp_gamma)
        # print("addend_sum_tmp denominator", denominator*decay)
        # print("--> addend for index", index, "is:", addend_sum_tmp)
        addend_sum += numerator / denominator
    # addend_sum /= decay  TODO: rm
    # print("addend_sum", addend_sum)  # so far: unparallelized

    int_part = 0
    for l in range(len(history) - 1):
        tl_minus_tj = history[l] - history[:l + 1]
        # print("times", ti_minus_tj)
        tlplus1_minus_tj = history[l + 1] - history[:l + 1]
        # print("times+1", tiplus1_minus_tj)
        exp_sigma = np.exp(-sigma * tl_minus_tj)
        exp_sigma_plus1 = np.exp(-sigma * tlplus1_minus_tj)
        exp_gamma = np.exp(-gamma * tl_minus_tj)
        exp_gamma_plus1 = np.exp(-gamma * tlplus1_minus_tj)
        int_part += (n - (l + 1)) / n * (
            1 / gms**2 *
            np.sum(
                exp_sigma - exp_sigma_plus1
            )
            +
            1 / gms *
            np.sum(
                -tl_minus_tj * exp_sigma + tlplus1_minus_tj * exp_sigma_plus1
            )
            -
            1 / gms**2 *
            np.sum(exp_gamma - exp_gamma_plus1)
        )
    int_part *= beta
    # print("addend_int FINISH:", int_part)
    return addend_sum - int_part


def dllf_dgamma_sigma_neq_gamma(beta, sigma, gamma, n, history,
                                sum_less_equal=True):
    """
    Parameters
    ----------
    beta : float
        See corresponding argument in :func:`exp_intensity`.
    sigma : float
        See corresponding argument in :func:`exp_intensity`.
    gamma : float
        See corresponding argument in :func:`exp_intensity`.
    n : float
        See corresponding argument in :func:`exp_intensity`.
    history : np.array
        See corresponding argument in :func:`exp_intensity`.
    sum_less_equal : bool, default: True
        See corresponding argument in :func:`exp_intensity`.

    Returns
    -------
    derivative_wrt_gamma : float
        The derivative (w.r.t. the gamma parameter) of the log-likelihood
        function given the `history` and evaluated at the parameters
        `beta`, `sigma`, `gammma`, and `n`.
    """
    his_after_zero = history[history > 0]
    gms = float(gamma - sigma)
    addend_sum = -np.count_nonzero(history) / gms

    for index, event_time in enumerate(his_after_zero):
        if sum_less_equal:
            ti_minus_tj = event_time - history[history <= event_time]
        else:
            ti_minus_tj = event_time - history[history < event_time]

        exp_sigma = np.exp(-sigma * ti_minus_tj)
        exp_gamma = np.exp(-gamma * ti_minus_tj)

        numerator = np.sum(-ti_minus_tj * exp_gamma)
        # print("addend_sum_tmp numerator:", addend_sum_tmp)
        denominator = np.sum(exp_sigma - exp_gamma)
        # print("addend_sum_tmp denominator", denominator*decay)
        # print("--> addend for index", index, "is:", addend_sum_tmp)
        addend_sum -= numerator / denominator
    # print("addend_sum", addend_sum)  # so far: unparallelized

    int_part = 0
    for l in range(len(history) - 1):
        tl_minus_tj = history[l] - history[:l + 1]
        # print("times", ti_minus_tj)
        tlplus1_minus_tj = history[l + 1] - history[:l + 1]
        # print("times+1", tiplus1_minus_tj)
        exp_sigma = np.exp(-sigma * tl_minus_tj)
        exp_sigma_plus1 = np.exp(-sigma * tlplus1_minus_tj)
        exp_gamma = np.exp(-gamma * tl_minus_tj)
        exp_gamma_plus1 = np.exp(-gamma * tlplus1_minus_tj)
        int_part += (n - (l + 1)) / n * (
            -1 / gms**2 *
            np.sum(
                exp_sigma - exp_sigma_plus1
            )
            -
            sigma * (sigma - 2 * gamma) / (gamma * gms)**2 *
            np.sum(exp_gamma - exp_gamma_plus1)
            -
            sigma / (gamma * gms) *
            np.sum(
                tlplus1_minus_tj * exp_gamma_plus1 - tl_minus_tj * exp_gamma
            )
        )
    int_part *= beta
    # print("addend_int FINISH:", int_part)
    return addend_sum - int_part


def dllf_dgamma_sigma_eq_gamma(beta, gamma, n, history, sum_less_equal=True):
    """
    Parameters
    ----------
    beta : float
        See corresponding argument in :func:`exp_intensity`.
    gamma : float
        See corresponding argument in :func:`exp_intensity`.
    n : float
        See corresponding argument in :func:`exp_intensity`.
    history : np.array
        See corresponding argument in :func:`exp_intensity`.
    sum_less_equal : bool, default: True
        See corresponding argument in :func:`exp_intensity`.

    Returns
    -------
    derivative_wrt_gamma : float
        The derivative (w.r.t. the gamma parameter) of the log-likelihood
        function given the `history` and evaluated at the parameters
        `beta`, `gammma`, and `n`.
    """
    # print("gamma (eq)", gamma)
    his_after_zero = history[history > 0]
    addend_sum = np.count_nonzero(history) / gamma

    for index, event_time in enumerate(his_after_zero):
        if sum_less_equal:
            ti_minus_tj = event_time - history[history <= event_time]
        else:
            ti_minus_tj = event_time - history[history < event_time]

        exp_gamma = np.exp(-gamma * ti_minus_tj)

        numerator = np.sum(ti_minus_tj**2 * exp_gamma)
        # print("addend_sum_tmp numerator:", addend_sum_tmp)
        denominator = np.sum(ti_minus_tj * exp_gamma)
        # print("addend_sum_tmp denominator", denominator*decay)
        # print("--> addend for index", index, "is:", addend_sum_tmp)
        addend_sum -= numerator / denominator
    # print("addend_sum", addend_sum)  # so far: unparallelized

    int_part = 0
    for l in range(len(history) - 1):
        tl_minus_tj = history[l] - history[:l + 1]
        tlplus1_minus_tj = history[l + 1] - history[:l + 1]

        g_tl_minus_tj = gamma * tl_minus_tj
        g_tlplus1_minus_tj = gamma * tlplus1_minus_tj

        int_part += (n - (l + 1)) / n * np.sum(
            np.exp(-g_tl_minus_tj)
            *
            (
                    -g_tl_minus_tj * (1 + g_tl_minus_tj) - 1
            )
            -
            np.exp(-g_tlplus1_minus_tj)
            *
            (
                    -g_tlplus1_minus_tj * (1 + g_tlplus1_minus_tj) - 1
            )
        )
    int_part *= beta / gamma**2
    # print("addend_int FINISH:", int_part)
    return addend_sum - int_part


def dllf_dn_sigma_neq_gamma(beta, sigma, gamma, n, history,
                            sum_less_equal=True):
    """
    Parameters
    ----------
    beta : float
        See corresponding argument in :func:`exp_intensity`.
    sigma : float
        See corresponding argument in :func:`exp_intensity`.
    gamma : float
        See corresponding argument in :func:`exp_intensity`.
    n : float
        See corresponding argument in :func:`exp_intensity`.
    history : np.array
        See corresponding argument in :func:`exp_intensity`.
    sum_less_equal : bool, default: True
        See corresponding argument in :func:`exp_intensity`.

    Returns
    -------
    derivative_wrt_n : float
        The derivative (w.r.t. the `n` parameter) of the log-likelihood
        function given the `history` and evaluated at the parameters
        `beta`, `sigma`, `gammma`, and `n`.
    """
    # print("we divide\n",
    #       np.arange(1, len(history) + 1),
    #       "\nby\n",
    #       (n * (n - np.arange(1, len(history) + 1))))
    # print(np.sum(
    #     np.arange(1, len(history) - sum(history <= 0) + 1) /
    #     (n - np.arange(1, len(history) - sum(history <= 0) + 1))) / n
    # )
    time_indices_gt_zero = np.arange(np.count_nonzero(history <= 0) + 1,  # missing in R code: +1
                                     len(history) + 1)  # added in R code: - sum(history <= 0))
    addend_sum = np.sum(
        time_indices_gt_zero / (n - time_indices_gt_zero)
    ) / n
    # print("addend_sum:", addend_sum)

    int_part = 0
    gms = gamma - sigma
    for l in range(len(history) - 1):
        tl_minus_tj = history[l] - history[:l + 1]
        tlplus1_minus_tj = history[l + 1] - history[:l + 1]
        int_part += (l + 1) * (
            np.sum(
                np.exp(-sigma * tl_minus_tj)
                -
                np.exp(-sigma * tlplus1_minus_tj)
            ) / sigma
            -
            np.sum(
                np.exp(-gamma * tl_minus_tj)
                -
                np.exp(-gamma * tlplus1_minus_tj)
            ) / gamma
        )
    int_part *= beta * sigma / gms / n**2
    # print("addend_int FINISH:", int_part)
    return addend_sum - int_part


def dllf_dn_sigma_eq_gamma(beta, gamma, n, history, sum_less_equal=True):
    """
    Parameters
    ----------
    beta : float
        See corresponding argument in :func:`exp_intensity`.
    gamma : float
        See corresponding argument in :func:`exp_intensity`.
    n : float
        See corresponding argument in :func:`exp_intensity`.
    history : np.array
        See corresponding argument in :func:`exp_intensity`.
    sum_less_equal : bool, default: True
        See corresponding argument in :func:`exp_intensity`.

    Returns
    -------
    derivative_wrt_n : float
        The derivative (w.r.t. the `n` parameter) of the log-likelihood
        function given the `history` and evaluated at the parameters
        `beta`, `gammma`, and `n`.
    """
    # print("we divide\n",
    #       np.arange(1, len(history) + 1),
    #       "\nby\n",
    #       (n * (n - np.arange(1, len(history) + 1))))
    # print(np.sum(
    #     np.arange(1, len(history) - sum(history <= 0) + 1) /
    #     (n - np.arange(1, len(history) - sum(history <= 0) + 1))) / n
    # )
    time_indices_gt_zero = np.arange(np.count_nonzero(history <= 0) + 1,  # missing in R code: +1
                                     len(history) + 1)  # added in R code: - sum(history <= 0))
    addend_sum = np.sum(
        time_indices_gt_zero / (n - time_indices_gt_zero)
    ) / n
    # print("addend_sum:", addend_sum)

    int_part = 0
    for l in range(len(history) - 1):
        tl_minus_tj = history[l] - history[:l + 1]
        tlplus1_minus_tj = history[l + 1] - history[:l + 1]
        mg_tl_minus_tj = -gamma * tl_minus_tj
        mg_tlplus1_minus_tj = -gamma * tlplus1_minus_tj
        int_part += (l + 1) * np.sum(
                np.exp(mg_tlplus1_minus_tj) * (mg_tlplus1_minus_tj - 1)
                -
                np.exp(mg_tl_minus_tj) * (mg_tl_minus_tj - 1)
        )
    int_part *= beta / gamma / n**2
    # print("addend_int FINISH:", int_part)
    return addend_sum - int_part


def llf_gradient(beta, sigma, gamma, n, history, sum_less_equal=True):
    if sigma is None or sigma == gamma:
        gradient = np.empty(3)
        gradient[0] = dllf_dbeta_sigma_eq_gamma(
            beta=beta, gamma=gamma, n=n, history=history,
            sum_less_equal=sum_less_equal
        )
        gradient[1] = dllf_dgamma_sigma_eq_gamma(
            beta=beta, gamma=gamma, n=n, history=history,
            sum_less_equal=sum_less_equal
        )
        gradient[2] = dllf_dn_sigma_eq_gamma(
            beta=beta, gamma=gamma, n=n, history=history,
            sum_less_equal=sum_less_equal
        )

    else:
        gradient = np.empty(4)
        gradient[0] = dllf_dbeta_sigma_neq_gamma(
            beta=beta, sigma=sigma, gamma=gamma, n=n, history=history,
            sum_less_equal=sum_less_equal
        )
        gradient[1] = dllf_dsigma_sigma_neq_gamma(
            beta=beta, sigma=sigma, gamma=gamma, n=n, history=history,
            sum_less_equal=sum_less_equal
        )
        gradient[2] = dllf_dgamma_sigma_neq_gamma(
            beta=beta, sigma=sigma, gamma=gamma, n=n, history=history,
            sum_less_equal=sum_less_equal
        )
        gradient[3] = dllf_dn_sigma_neq_gamma(
            beta=beta, sigma=sigma, gamma=gamma, n=n, history=history,
            sum_less_equal=sum_less_equal
        )
        # print("beta", round(beta, 2),
        #       "sigma", round(sigma, 2),
        #       "gamma", round(gamma, 2),
        #       "n", round(n, 2))
        # print("gamma", gamma)
        # print("gradient: ", gradient)
    return gradient


# def to_seir(scale, decay, n):  # todo
#     """
#     Return the parameters of the equivalent SIR model.
#
#     Parameters
#     ----------
#     scale : float
#         See the corresponding parameters in :meth:`exp_intensity` for more
#         information.
#     decay : float
#         See the corresponding parameters in :meth:`exp_intensity` for more
#         information.
#     n : int
#         See the corresponding parameters in :meth:`exp_intensity` for more
#         information.
#
#     Returns
#     -------
#     sir_parameters : dict
#         Dictionary containing the corresponding SIR parameters n, i_0, beta
#         (infection rate), and gamma (recovery rate).
#
#     Examples
#     --------
#     >>> obtained = HawkesN.to_sir(scale=5, decay=0.2, n=400)
#     >>> desired = {"n": 400, "i_0": 1, "beta": 1.0, "gamma": 0.2}
#     >>> obtained == desired
#     True
#     """
#     i_0 = 1  # the HawkesN process is assumed to have 1 "immigrant" at t=0
#     beta = scale * decay
#     gamma = decay
#     return {"n": n, "i_0": i_0, "beta": beta, "gamma": gamma}


# def size_distribution(scale=5, decay=0.2, n=400, history=None,
#                           transition_matrix=None):
#     sir_params = to_sir(scale=scale, decay=decay, n=n)
# 
#     if history is None:
#         return StochasticSEIR.size_distribution(
#             **seir_params, transition_matrix=transition_matrix)
#     else:
#         max_time = history[-2]

