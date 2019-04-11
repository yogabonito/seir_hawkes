import numbers

from scipy.optimize import fmin_l_bfgs_b
import numpy as np

from .sir_stochastic import StochasticSIR


def exp_intensity(scale, decay, history, sum_less_equal=True):
    """
    Calculate the (exponential) intensity of a HawkesN process.

    Parameters
    ----------
    scale : float
        Scale parameter of the exponential intensity.
    decay : float
        Decay parameter of the exponential intensity.
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
        result *= scale * decay
        return result

    return exp_intensity_


def llf(scale, decay, history, sum_less_equal=True):
    """

    Parameters
    ----------
    scale : float
        See corresponding argument in :meth:`self.exp_intensity`.
    decay : float
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
    intensity = exp_intensity(scale, decay, history,
                              sum_less_equal=sum_less_equal)
    sum_part = np.sum(np.log(intensity(history[history > 0])))

    int_part = 0
    for i in range(len(history) - 1):
        int_part += np.sum(
            np.exp(-decay * (history[i] - history[:i + 1]))
            -
            np.exp(-decay * (history[i + 1] - history[:i + 1]))
        )
    int_part *= scale
    print("sum:", sum_part)
    print("integral:", int_part)
    print("*** llf:", sum_part - int_part)
    return sum_part - int_part


def fit(history, scale_start=None, decay_start=None):
    """
    Parameters
    ----------
    history : np.array
        1-dimensional array containing the event times in ascending order.
    scale_start : float
        Starting value for the likelihood optimization.
    decay_start : float
        Starting value for the likelihood optimization.

    Returns
    -------
    TODO
    References
    ----------
    This method uses the L-BFGS algorithm (see [1]_).
    .. [1] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    """

    def negative_llf(scale_decay):
        """
        Parameters
        ----------
        scale_decay : np.array (shape (3))
            Values for the scale and decay parameter and the parameter N
            a single array.

        Returns
        -------
        neg_llf : float
            The negative log-likelihood.
        """
        scale, decay = tuple(scale_decay)
        return -llf(scale=scale, decay=decay, history=history)

    def negative_llf_gradient(scale_decay):
        scale, decay = tuple(scale_decay)
        return -llf_gradient(scale=scale, decay=decay, history=history)

    eps = np.finfo(float).eps

    return fmin_l_bfgs_b(
        func=negative_llf,  # minimize this
        x0=np.array([scale_start, decay_start]),  # initial guess
        fprime=negative_llf_gradient,
        bounds=[(eps, None), (eps, None)],
        iprint=1
    )


def dllf_dscale(scale, decay, history, sum_less_equal=True):
    """
    Parameters
    ----------
    scale
    decay
    history
    sum_less_equal : bool
        This argument does not affect the derivative w.r.t. the scale
        parameter. Thus, the return value does not depend on it.

    Returns
    -------
    derivative_wrt_scale : float
        The derivative (w.r.t. the scale parameter) of the log-likelihood
        function given the `history` and evaluated at the parameters
        `scale` and `decay`.
    """
    sum_part = np.count_nonzero(history) / scale
    # print("#" * 80, "\n", "SCALE", sep="")
    # print("#" * 80)
    # print("addend sum:", addend_sum)
    # print("#" * 80)
    # print("#" * 80)
    int_part = 0
    for i in range(len(history) - 1):
        int_part += np.sum(
            np.exp(-decay * (history[i] - history[:i + 1]))
            -
            np.exp(-decay * (history[i + 1] - history[:i + 1]))
        )
    return sum_part - int_part


def dllf_ddecay(scale, decay, history, sum_less_equal=True):
    """

    Parameters
    ----------
    scale
    decay
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
        `scale` and `decay`.
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
    print("addend_sum", addend_sum)  # so far: unparallelized

    int_part = 0
    for i in range(len(history) - 1):
        ti_minus_tj = history[i] - history[:i + 1]
        # print("times", ti_minus_tj)
        tiplus1_minus_tj = history[i + 1] - history[:i + 1]
        # print("times+1", tiplus1_minus_tj)
        int_part += np.sum(
            - ti_minus_tj * np.exp(-decay * ti_minus_tj)
            + tiplus1_minus_tj * np.exp(-decay * tiplus1_minus_tj)
        )
    int_part *= scale
    print("addend_int FINISH:", int_part)
    return addend_sum - int_part


def llf_gradient(scale, decay, history):
    gradient = np.empty(2)
    gradient[0] = dllf_dscale(scale=scale, decay=decay, history=history)
    gradient[1] = dllf_ddecay(scale=scale, decay=decay, history=history)
    return gradient
