import numbers

from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import matplotlib.pyplot as plt

from .sir_stochastic import StochasticSIR


def exp_intensity(scale, decay, n, history, sum_less_equal=True):
    """
    Calculate the (exponential) intensity of a HawkesN process.

    Parameters
    ----------
    scale : float
        Scale parameter of the exponential intensity.
    decay : float
        Decay parameter of the exponential intensity.
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
        if isinstance(t, numbers.Number):
            t = np.array([t])
        result = np.empty(t.shape)
        for index, time in enumerate(t):
            if sum_less_equal:
                history_until_t = history[history <= time]
            else:
                history_until_t = history[history < time]
            result[index] = np.sum(np.exp(
                -decay * (time - history_until_t)
            ))
            result[index] *= (1 - np.count_nonzero(history <= time)/n)
            # print("intensity at index", index, "is", result[index]*scale*decay)
        result *= scale * decay
        # if np.isnan(result).any():
            # print("NAN FOR:")
            # print("scale", scale)
            # print("decay", decay)
            # print("n", n)
            # print("history", history[:5])
        return result
    return exp_intensity_


def plot_exp_intensity(t_max, intensity=None, scale=None,
                       decay=None, n=None, history=None, width=5.51,
                       height=4, n_xticks=6, step=.01,
                       fname=None, **kwargs):
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
    scale : float or None
        Scale parameter of the exponential intensity. Ignored if
        exp_intensity is provided as argument.
    decay : float or None
        Decay parameter of the exponential intensity. Ignored if
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
        intensity = exp_intensity(scale, decay, n, history, **kwargs)

    t = np.arange(0, t_max, step)

    plt.figure(dpi=300, figsize=(width, height))
    plt.plot(t, intensity(t))

    plt.xlabel("$t$")
    plt.xlim(0, t_max)
    plt.xticks(np.linspace(0, t_max, n_xticks))

    plt.ylabel("Intensity")

    plt.grid()

    title = "Intensity of a HawkesN process"
    if history is not None and scale is not None and decay is not None \
            and n is not None:
        title += " with event history \{" \
                 + ",".join(str(i) for i in history[:4]) \
                 + (", ..." if len(history) > 4 else "") \
                 + "\} \nand parameters: scale=" + str(scale) \
                 + ", decay=" + str(decay) + ", $N$=" + str(n)
    title += "."
    plt.title(title)

    if fname is not None:
        plt.savefig(fname + ".pdf")


def llf(scale, decay, n, history, sum_less_equal=True):
    """

    Parameters
    ----------
    scale : float
        See corresponding argument in :meth:`self.exp_intensity`.
    decay : float
        See corresponding argument in :meth:`self.exp_intensity`.
    n : float
        See corresponding argument in :meth:`self.exp_intensity`.
    history : np.array
        See corresponding argument in :meth:`self.exp_intensity`.
    sum_less_equal : bool, default: True
        Used as argument in :meth:`self.exp_intensity`.


    Returns
    -------
    llf : numpy.float64
        The log-likelihood for the parameters passed as arguments.
    """
    intensity = exp_intensity(scale, decay, n, history,
                                  sum_less_equal=sum_less_equal)
    try:
        sum_part = np.sum(np.log(intensity(history[history > 0])))
    except Exception as e:
        print("his", history)
        print("his>0", history[history>0])
        print("int", intensity(history[history > 0]))
        print("log", np.log(intensity(history[history > 0])))
        print("sum", np.sum(np.log(intensity(history[history > 0]))))
        raise Exception("truth val...")

    int_part = 0
    for i in range(len(history) - 1):
        int_part += (n - (i + 1)) / n * np.sum(
            np.exp(-decay * (history[i] - history[:i + 1]))
            -
            np.exp(-decay * (history[i + 1] - history[:i + 1]))
        )
    int_part *= scale
    # print("sum:", sum_part)
    # print("integral:", int_part)
    # print("*** llf:", sum_part - int_part)
    return sum_part - int_part


def fit(history, scale_start=None, decay_start=None, n_start=None,
        scale_decay_fixed=False):
    """
    Parameters
    ----------
    history : np.array
        1-dimensional array containing the event times in ascending order.
    scale_start : float
        Starting value for the likelihood optimization.
    decay_start : float
        Starting value for the likelihood optimization.
    n_start : float
        Starting value for the likelihood optimization.
    scale_decay_fixed : bool, default: False
        If True, `scale` and `decay` are considered to be fixed and only
        :math:`N` is fitted. Otherwise, `scale` and `decay` are fitted together
        with :math:`N`.

    Returns
    -------
    TODO
    References
    ----------
    This method uses the L-BFGS algorithm (see [1]_).
    .. [1] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    """

    if scale_decay_fixed and (scale_start is None or decay_start is None):
        raise Exception("If scale and decay are fixed, their values must be "
                        "provided!")

    def negative_llf(scale_decay_n):
        """
        Parameters
        ----------
        scale_decay_n : np.array (shape (3))
            Values for the scale and decay parameter and the parameter N
            a single array.

        Returns
        -------
        neg_llf : float
            The negative log-likelihood.
        """
        scale, decay, n = tuple(scale_decay_n)
        return -llf(scale=scale, decay=decay, n=n, history=history)

    def negative_llf_separate_params(n, scale, decay):
        """
        Same as :func:`negative_llf` but taking the parameters `n`, `scale`,
        and `decay` as separate arguments. This makes the function suitable for
        likelihood maximization in only one parameter (`n`) with fixed values
        for `scale` and `decay`.
        """
        return -llf(scale=scale, decay=decay, n=n, history=history)

    def negative_llf_gradient(scale_decay_n):
        scale, decay, n = tuple(scale_decay_n)
        return -llf_gradient(scale=scale, decay=decay, n=n,
                                 history=history)

    def negative_llf_gradient_separate_params(n, scale, decay):
        return -dllf_dn(scale=scale, decay=decay, n=n, history=history)

    eps = np.finfo(float).eps

    if scale_decay_fixed:
        return fmin_l_bfgs_b(
            func=negative_llf_separate_params,  # minimize this
            x0=np.array([n_start]),  # initial guess
            args=(scale_start, decay_start),  # additional args to func&fprime
            fprime=negative_llf_gradient_separate_params,
            bounds=[(len(history), None)],
            iprint=0
        )

    else:
        return fmin_l_bfgs_b(
            func=negative_llf,  # minimize this
            x0=np.array([scale_start, decay_start, n_start]),  # initial guess
            fprime=negative_llf_gradient,
            bounds=[(eps, None), (eps, None), (len(history), None)],
            iprint=0
        )


def dllf_dscale(scale, decay, n, history, sum_less_equal=True):
    """
    Parameters
    ----------
    scale
    decay
    n
    history
    sum_less_equal : bool
        This argument does not affect the derivative w.r.t. the scale
        parameter. Thus, the return value does not depend on it.

    Returns
    -------
    derivative_wrt_scale : float
        The derivative (w.r.t. the scale parameter) of the log-likelihood
        function given the `history` and evaluated at the parameters
        `scale`, `decay`, and `n`.
    """
    sum_part = np.count_nonzero(history) / scale
    # print("#" * 80, "\n", "SCALE", sep="")
    # print("#" * 80)
    # print("addend sum:", addend_sum)
    # print("#" * 80)
    # print("#" * 80)
    int_part = 0
    for i in range(len(history) - 1):
        int_part += (n - (i + 1)) / n * np.sum(
            np.exp(-decay * (history[i] - history[:i + 1]))
            -
            np.exp(-decay * (history[i + 1] - history[:i + 1]))
        )
    return sum_part - int_part


def dllf_ddecay(scale, decay, n, history, sum_less_equal=True):
    """

    Parameters
    ----------
    scale
    decay
    n
    history
    sum_less_equal : bool, default: True
        The calculation involves the evaluation of the process' intensity
        at event times. This argument causes the calculations to be
        executed as described in the docstring of
        :meth:`self.exp_intensity`.

    Returns
    -------
    derivative_wrt_scale : float
        The derivative (w.r.t. the decay parameter) of the log-likelihood
        function given the `history` and evaluated at the parameters
        `scale`, `decay`, and `n`.
    """
    his_after_zero = history[history > 0]

    addend_sum = 0
    for index, event_time in enumerate(his_after_zero):
        if sum_less_equal:
            ti_minus_tj = event_time - history[history <= event_time]
        else:
            ti_minus_tj = event_time - history[history < event_time]
        exp = np.exp(-decay * ti_minus_tj)
        addend_sum_tmp = np.sum((1 - decay * ti_minus_tj) * exp)
        # print("addend_sum_tmp numerator:", addend_sum_tmp)
        denominator = np.sum(exp)
        addend_sum_tmp /= denominator
        # print("addend_sum_tmp denominator", denominator*decay)
        # print("--> addend for index", index, "is:", addend_sum_tmp)
        addend_sum += addend_sum_tmp
    addend_sum /= decay
    # print("addend_sum", addend_sum)  # so far: unparallelized

    int_part = 0
    for i in range(len(history) - 1):
        ti_minus_tj = history[i] - history[:i + 1]
        # print("times", ti_minus_tj)
        tiplus1_minus_tj = history[i + 1] - history[:i + 1]
        # print("times+1", tiplus1_minus_tj)
        int_part += (n - (i + 1)) / n * np.sum(
            - ti_minus_tj * np.exp(-decay * ti_minus_tj)
            + tiplus1_minus_tj * np.exp(-decay * tiplus1_minus_tj)
        )
    int_part *= scale
    # print("addend_int FINISH:", int_part)
    return addend_sum - int_part


def dllf_dn(scale, decay, n, history, sum_less_equal=True):
    """

    Parameters
    ----------
    scale
    decay
    n
    history
    sum_less_equal : bool
        This argument does not affect the derivative w.r.t. the scale
        parameter. Thus, the return value does not depend on it.

    Returns
    -------
    derivative_wrt_scale : float
        The derivative (w.r.t. the population size) of the log-likelihood
        function given the `history` and evaluated at the parameters
        `scale`, `decay`, and `n`.
    """
    # print("we devide\n",
    #      np.arange(1, len(history) + 1),
    #      "\nby\n",
    #      (n * (n - np.arange(1, len(history) + 1))))
    # print(np.sum(
    #    np.arange(1, len(history) - sum(history <= 0) + 1) /
    #    (n - np.arange(1, len(history) - sum(history <= 0) + 1))) / n
    #)
    time_indices_gt_zero = np.arange(np.count_nonzero(history <= 0) + 1,  # missing in R code: +1
                                     len(history) + 1)  # added in R code: - sum(history <= 0))
    addend_sum = np.sum(
        time_indices_gt_zero / (n - time_indices_gt_zero)
    ) / n
    # print("addend_sum:", addend_sum)

    int_part = 0
    for i in range(len(history) - 1):
        ti_minus_tj = history[i] - history[:i + 1]
        tiplus1_minus_tj = history[i + 1] - history[:i + 1]
        int_part += (i + 1) * np.sum(
            np.exp(-decay * ti_minus_tj)
            -
            np.exp(-decay * tiplus1_minus_tj)
        )
    int_part *= scale / n**2
    # print("addend_int FINISH:", int_part)
    return addend_sum - int_part


def llf_gradient(scale, decay, n, history):
    gradient = np.empty(3)
    gradient[0] = dllf_dscale(scale=scale, decay=decay, n=n,
                                  history=history)
    gradient[1] = dllf_ddecay(scale=scale, decay=decay, n=n,
                                  history=history)
    gradient[2] = dllf_dn(scale=scale, decay=decay, n=n,
                              history=history)
    return gradient


def to_sir(scale, decay, n):
    """
    Return the parameters of the equivalent SIR model.

    Parameters
    ----------
    scale : float
        See the corresponding parameters in :meth:`exp_intensity` for more
        information.
    decay : float
        See the corresponding parameters in :meth:`exp_intensity` for more
        information.
    n : int
        See the corresponding parameters in :meth:`exp_intensity` for more
        information.

    Returns
    -------
    sir_parameters : dict
        Dictionary containing the corresponding SIR parameters n, i_0, beta
        (infection rate), and gamma (recovery rate).

    Examples
    --------
    >>> obtained = HawkesN.to_sir(scale=5, decay=0.2, n=400)
    >>> desired = {"n": 400, "i_0": 1, "beta": 1.0, "gamma": 0.2}
    >>> obtained == desired
    True
    """
    i_0 = 1  # the HawkesN process is assumed to have 1 "immigrant" at t=0
    beta = scale * decay
    gamma = decay
    return {"n": n, "i_0": i_0, "beta": beta, "gamma": gamma}


