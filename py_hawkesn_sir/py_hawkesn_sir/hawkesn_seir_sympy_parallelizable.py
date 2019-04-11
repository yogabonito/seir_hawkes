# from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from sympy import derive_by_array, exp, lambdify, log, Piecewise, symbols


def exp_intensity_sigma_neq_gamma(history, sum_less_equal=True):
    """
    Calculate the (exponential) intensity of a (SEIR-)HawkesN process
    symbolically.

    Parameters
    ----------
    sum_less_equal : bool, default: True
        If True, we sum over all event times <= time t. Otherwise, we sum
        over all event times < time t.

    Returns
    -------
    exp_intensity_ : sympy.core.mul.Mul
        A sympy expression containing the symbols beta, sigma, gamma, n,
        and t.
    """
    beta, sigma, gamma, n, t = symbols("beta sigma gamma n t")
    events_until_t = sum(
        [Piecewise((1, h <= t), (0, True)) for h in history]
    )

    return (1 - events_until_t / n) * (beta * sigma / (gamma-sigma)) * sum(
        [Piecewise(
            (
                exp(-sigma * (t - h)) - exp(-gamma * (t - h)),
                h <= t if sum_less_equal else h < t
             ),
            (0, True)
         ) for h in history])


def exp_intensity_sigma_eq_gamma(history, sum_less_equal=True):
    """
    Calculate the (exponential) intensity of a (SEIR-)HawkesN process
    symbolically.

    Parameters
    ----------
    sum_less_equal : bool, default: True
        If True, we sum over all event times <= time t. Otherwise, we sum
        over all event times < time t.

    Returns
    -------
    exp_intensity_ : sympy.core.mul.Mul
        A sympy expression containing the symbols beta, gamma, n,
        and t. The symbol sigma is not contained as sigma=gamma holds in
        the case considered by this function.
    """
    beta, gamma, n, t = symbols("beta gamma n t")
    events_until_t = sum(
        [Piecewise((1, h <= t), (0, True)) for h in history]
    )

    return (1 - events_until_t / n) * beta * gamma * sum(
        [Piecewise(
            (
                (t - h) * exp(-gamma * (t - h)),
                h <= t if sum_less_equal else h < t
            ),
            (0, True)
         ) for h in history])


def plot_exp_intensity(history, t_max, beta, sigma, gamma, n, step=0.01,
                       width=5.51, height=4, n_xticks=6, fname=None,
                       sum_less_equal=True):
    """
    Plot (or save the plot of) the exponential intensity function from t=0
    until t=t_max.

    Parameters
    ----------
    t_max : float
        Define the time horizon of the plot. The time axis will contain
        values from 0 to t_max.
    beta : float
        Parameter beta of the SEIR model.
    sigma : float or None
        Parameter sigma of the SEIR model. If None, then sigma=gamma is
        assumed.
    gamma : float
        Parameter gamma of the SEIR model.
    n : int
        Population size.
    step : float, default: 0.01
        Interval on the x-axis between two successive points.
    width : float, default: 5.51
        Width of the plot.
    height : float, default: 4.0
        Height of the plot.
    n_xticks : int (must be non-negative)
        Number of ticks on the time axis.
    fname : str or None
        Name (without extension) of the file the plot is saved to. If
        `None`, the plot is not saved.
    sum_less_equal : bool
        This arg is used in :func:`exp_intensity`.
    """
    if sigma is None:
        sigma = gamma
    subs_list = [("beta", beta), ("sigma", sigma), ("gamma", gamma),
                 ("n", n)]
    if sigma == gamma:
        exp_intensity = exp_intensity_sigma_eq_gamma(
            history, sum_less_equal=sum_less_equal).subs(subs_list)
    else:
        exp_intensity = exp_intensity_sigma_neq_gamma(
            history, sum_less_equal=sum_less_equal).subs(subs_list)
    exp_intensity = lambdify("t", exp_intensity)

    time = np.arange(0, t_max, step)

    plt.figure(dpi=300, figsize=(width, height))
    plt.plot(time, exp_intensity(time))

    plt.xlabel("$t$")
    plt.xlim(0, t_max)
    plt.xticks(np.linspace(0, t_max, n_xticks))

    plt.ylabel("Intensity")

    plt.grid()

    title = "Intensity of a HawkesN process"
    if history is not None and beta is not None and sigma is not None \
            and gamma is not None and n is not None:
        title += " with event history \{" \
                 + ",".join(str(i) for i in history[:4]) \
                 + (", ..." if len(history) > 4 else "") \
                 + "\} \nand parameters: beta=" + str(beta) \
                 + ", sigma=" + str(sigma) + ", gamma=" + str(gamma) \
                 + ", $N$=" + str(n)
    title += "."
    plt.title(title)

    if fname is not None:
        plt.savefig(fname + ".pdf")


def llf_sigma_neq_gamma(history, sum_less_equal=True):
    """
    Parameters
    ----------
    sum_less_equal : bool, default: True
        This arg is used in :func:`exp_intensity_sigma_neq_gamma`.

    Returns
    -------
    llf : sympy.core.add.Add
        The log-likelihood function as symbolic expression (containing the
        symbols `beta`, `sigma`, `gamma`, and `n`).
    """
    beta, sigma, gamma, n = symbols("beta sigma gamma n")
    intensity = exp_intensity_sigma_neq_gamma(history, sum_less_equal)

    # for h in self.his:
    #     print("intensity at", h, "is:", intensity.subs("t", h))
    first_event = len(history) - sum(1 for t in history if t > 0)
    his_pos = history[first_event:]
    addend_sum = sum(log(intensity.subs("t", h)) for h in his_pos)
    # print("SUM PART", addend_sum.subs([("scale", .5), ("decay", .5), ("n", 100)]))

    addend_int = (beta * sigma / (gamma-sigma)) * sum(
        (n - (i + 1)) / n * (
            (
                exp(-sigma * (history[i] - history[j]))
                -
                exp(-sigma * (history[i + 1] - history[j]))
            ) / sigma
            -
            (
                exp(-gamma * (history[i] - history[j]))
                -
                exp(-gamma * (history[i + 1] - history[j]))
            ) / gamma
        )
        for i in range(len(history)-1)
        for j in range(i+1))
    # print("INT PART", addend_int.subs([("scale", .5), ("decay", .5), ("n", 100)]))
    return addend_sum - addend_int


def llf_sigma_eq_gamma(history, sum_less_equal=True):
    """
    Parameters
    ----------
    sum_less_equal : bool, default: True
        This arg is used in :meth:`self.exp_intensity_sigma_eq_gamma`.

    Returns
    -------
    llf : sympy.core.add.Add
        The log-likelihood function as symbolic expression (containing the
        symbols `beta`, `gamma`, and `n`).
    """
    beta, gamma, n = symbols("beta gamma n")
    intensity = exp_intensity_sigma_eq_gamma(history, sum_less_equal)

    # for h in history:
    #     print("intensity at", h, "is:", intensity.subs("t", h))
    first_event = len(history) - sum(1 for t in history if t > 0)
    his_pos = history[first_event:]
    addend_sum = sum(log(intensity.subs("t", h)) for h in his_pos)
    # print("SUM PART", addend_sum.subs([("scale", .5), ("decay", .5), ("n", 100)]))

    addend_int = beta / gamma * sum(
        (n - (i + 1)) / n * (
            (
                exp(-gamma * (history[i] - history[j]))
                * (gamma * (history[i] - history[j]) + 1)
                -
                exp(-gamma * (history[i + 1] - history[j]))
                * (gamma * (history[i + 1] - history[j]) + 1)
            )
        )
        for i in range(len(history)-1)
        for j in range(i+1))
    # print("INT PART", addend_int.subs([("scale", .5), ("decay", .5), ("n", 100)]))
    return addend_sum - addend_int


def llf_gradient_sigma_neq_gamma(history, sum_less_equal=True):
    """
    Calculate the gradient of the log-likelihood function symbolically.

    Parameters
    ----------
    sum_less_equal : bool, default: True
        This arg is passed to :meth:`self.llf_sigma_eq_gamma`.

    Returns
    -------
    gradient : sympy.Array
        An array containing four entries. The first (second) [third]
        {fourth} entry is the derivative of the log-likelihood function
        w.r.t. beta (sigma) [gamma] {N} parameter.
    """
    beta, sigma, gamma, n = symbols("beta sigma gamma n")
    return derive_by_array(
        llf_sigma_neq_gamma(history, sum_less_equal),
        [beta, sigma, gamma, n]
    )

def llf_gradient_sigma_eq_gamma(history, sum_less_equal=True):
    """
    Calculate the gradient of the log-likelihood function symbolically.

    Parameters
    ----------
    sum_less_equal : bool, default: True
        This arg is passed to :meth:`self.llf_sigma_eq_gamma`.

    Returns
    -------
    gradient : sympy.Array
        An array containing four entries. The first [second] {third} entry
        is the derivative of the log-likelihood function w.r.t. beta
        [gamma] {N} parameter. There is no derivative w.r.t. sigma as it is
        considered equal to gamma in the case considered by this function.
    """
    beta, gamma, n = symbols("beta gamma n")
    return derive_by_array(
        llf_sigma_eq_gamma(history, sum_less_equal),
        [beta, gamma, n]
    )


# def fit(scale_start, decay_start, n_start):
#     """
#     Parameters
#     ----------
#     scale_start : float
#         Starting value for the likelihood maximization.
#     decay_start : float
#         Starting value for the likelihood maximization.
#     n_start : float
#         Starting value for the likelihood maximization.
#
#     Returns
#     -------
#     ...
#     """
#     llf_sym = self.llf()
#     llf_grad_sym = self.llf_gradient()
#     def negative_llf(scale_decay_n):
#         """
#         Parameters
#         ----------
#         scale_decay_n : np.array (shape (3))
#             Values for the scale and decay parameter and the parameter N
#             a single array.
#
#         Returns
#         -------
#         neg_llf : float
#             The negative log-likelihood.
#         """
#         result = llf_sym.subs([("scale", scale_decay_n[0]),
#                              ("decay", scale_decay_n[1]),
#                              ("n", scale_decay_n[2])])
#         print("llf", result)
#         return result
#
#     def negative_llf_gradient(scale_decay_n):
#         result = -llf_grad_sym.subs([("scale", scale_decay_n[0]),
#                                      ("decay", scale_decay_n[1]),
#                                      ("n", scale_decay_n[2])])
#         print("-grad:", result)
#         return np.array(result, dtype=np.float64)
#
#     eps = np.finfo(float).eps
#
#     return fmin_l_bfgs_b(
#         func=negative_llf,  # minimize this
#         x0=np.array([scale_start, decay_start, n_start]),  # initial guess
#         fprime=negative_llf_gradient,
#         bounds=[(eps, None), (eps, None), (len(self.his), None)],
#         iprint=101
#     )
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
        nllf = -llf_sigma_neq_gamma(history=history, sum_less_equal=False)
        return nllf.subs(
            [("beta", beta), ("sigma", sigma), ("gamma", gamma), ("n", n)])

    def negative_llf_separate_params(n, beta, sigma, gamma):
        """
        Same as :func:`negative_llf` but taking the parameters `n`, `beta`,
        `sigma`, and `gamma` as separate arguments. This makes the function
        suitable for likelihood maximization in only one parameter (`n`) with
        fixed values for `beta`, `sigma`, and `gamma`.
        """
        if sigma == gamma:
            sigma += 1e-7
        return -llf_sigma_neq_gamma(
            history=history, sum_less_equal=False).subs(
                [("beta", beta), ("sigma", sigma), ("gamma", gamma), ("n", n)])

    def negative_llf_gradient(beta_sigma_gamma_n):
        beta, sigma, gamma, n = tuple(beta_sigma_gamma_n)
        if sigma == gamma:
            sigma += 1e-7
        return -llf_gradient_sigma_neq_gamma(
            history=history, sum_less_equal=False).subs(
                [("beta", beta), ("sigma", sigma), ("gamma", gamma), ("n", n)])

    def negative_llf_gradient_separate_params(n, beta, sigma, gamma):
        if sigma == gamma:
            sigma += 1e-7
        return -llf_gradient_sigma_neq_gamma(
            history=history, sum_less_equal=False).subs(
                [("beta", beta),
                 ("sigma", sigma),
                 ("gamma", gamma),
                 ("n", n)])[-1]

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
            iprint=1
        )

