from math import log

from scipy.stats import expon
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd

from .util import get_index, int_pairs_leq_n_when_summed


def _rates_sir(state, beta, gamma, n):
    """
    Calculate the intensities of the cumulated infection process and the
    recovery process as well as birth/death rates.

    Parameters
    ----------
    state : dict or pd.Series
        Dictionary or pd.Series with the keys "s", "i", "r". The values
        associated with the keys are integers.
    beta : float
        Parameter beta of the SIR model.
    gamma : float
        Parameter gamma of the SIR model.
    n : int
        Parameter N of the SIR model.

    Returns
    -------
    rate_i : float
        Rate at which an infection occurs.
    rate_r : float
        Rate at which a recovery occurs.
    change_rate : float
        Sum of the other returned rates.
    """
    rate_i = beta * (state["s"] / n) * state["i"]
    rate_r = gamma * state["i"]
    change_rate = rate_i + rate_r
    return rate_i, rate_r, change_rate


def llf(observation, n, beta, gamma, verbose):
    """
    Compute the log-likelihood of the simulation specified by
    `simulation_index` given the parameters N, :math:`\\beta`, and
    :math:`\\gamma`.

    Parameters
    ----------
    observation : pandas.DataFrame
        DataFrame representing the observed SIR model. Columns are "s", "i",
        "r", and "t" for the compartments S, I, and R and the time,
        respectively.
    n : int or float
        Total number of individuals in the SIR model.
    beta : float
        Parameter beta in the SIR model. See corresponding argument in
        :meth:`simulate` for more information.
    gamma : float
        Parameter gamma in the SIR model. See corresponding argument in
        :meth:`simulate` for more information.
    verbose : bool
        If True, print debug output.

    Returns
    -------
    nllf : float
        The log-likelihood.

    Notes
    -----
    The main difference to the API of the R code is the missing I.0 parameter
    which is not used in the R code as well.
    """
    llf = 0
    state_new = observation.iloc[0]
    t_new = state_new["t"]
    for j in range(1, observation.shape[0]):
        if verbose:
            print(state_new)
        t_old = t_new
        state_old = state_new
        state_new = dict(observation.iloc[j])
        t_new = state_new["t"]
        rate_i, rate_r, change_rate = _rates_sir(state_old, beta, gamma, n)
        if change_rate == 0:
            break
        # expon.pdf(t_new - t_old, scale=1 / change_rate) gives the likelihood
        # of waiting the time we waited
        # NOTE: R code uses `rate=change_rate` while in Python it is
        # `scale=1/change_rate`
        llf += expon.logpdf(t_new - t_old, scale=1 / change_rate)
        # likelihood of observing the event we observed
        if _infection(state_old, state_new):
            llf += log(rate_i / change_rate)
        elif _recovery(state_old, state_new):
            llf += log(rate_r / change_rate)
    return llf


def negative_llf(n_beta_gamma, observation, verbose):
    """
    Wrapper for :func:`llf` taking n, beta, gamma as a single array and
    returning the *negative* log-likelihood.

    Parameters
    ----------
    n_beta_gamma : np.array (shape (3))
        Values for N, beta, and gamma provided in a single array.
    observation : pandas.DataFrame
        See corresponding parameter in :func:`llf`.
    verbose : bool
        If True, print arguments.

    Returns
    -------
    neg_llf : float
        The negative log-likelihood.
    """
    n, beta, gamma = tuple(n_beta_gamma)
    if verbose:
        print("  args n, beta, gamma:", n, beta, gamma)
    return -llf(observation, n, beta, gamma, verbose)


def negative_llf_separate_params(n, beta, gamma, observation, verbose):
    """
    Same as :func:`negative_llf`. The only difference is that the three
    parameters n, beta, and gamma are provided as separate arguments and not in
    a single array. This makes the function suitable for optimization purposes
    where beta and gamma are fixed.
    """
    if verbose:
        print("  args n, beta, gamma:", n, beta, gamma)
    return -llf(observation, n, beta, gamma, verbose)


def fit(observation, n_start=None, beta_start=0.1, gamma_start=0.2,
        verbose=False, observation_id=None, step_size=None,
        beta_gamma_fixed=False):
    """
    Parameters
    ----------
    observation : pandas.DataFrame
        See corresponding argument in :func:`llf`.
    n_start : int, float or None
        If int or float, it is the starting value for the likelihood
        optimization. If None, then the starting value will be the number
        of infections that has been observed in each data set.
    beta_start : float
        Starting value for the likelihood optimization.
    gamma_start : float
        Starting value for the likelihood optimization.
    verbose : bool, default: False
        If True, output is printed during the fitting.
    observation_id : int or None
        If int and verbose==True, then display the id of the observation.
    step_size : int, float, or numpy.array; optional
        If provided, then it is the step size for the L-BFGS algorithm. It can
        be provided as (one-dimensional) numpy array (of length 3) which
        contains the step sizes for :math:`N`, :math:`\\beta`, and
        :math:`\\gamma`.
    beta_gamma_fixed : bool, default: False
        If True, :math:`\\beta` and :math:`\\gamma` are considered to be fixed
        and only :math:`N` is fitted. Otherwise, :math:`\\beta` and
        :math:`\\gamma` are fitted together with :math:`N`.

    Returns
    -------
    result : tuple
        The result produced by the function :func:`fmin_l_bfgs_b` which
        implements the L-BFGS algorithm (see [1]_).

    References
    ----------
    This method uses the L-BFGS algorithm (see [1]_).
    .. [1] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
    """
    eps = np.finfo(float).eps

    if n_start is None:
        n_bounds = (observation.i.iloc[-1] + observation.r.iloc[-1], None)
        n_st = n_bounds[0] + 0.5  # + 0.5 to avoid start at bound
    else:
        n_st = n_start
        n_bounds = (eps, None)

    if verbose:
        if observation_id is not None:
            print("Fit observation number", observation_id)
        print("n_start is", n_start)
    
    if beta_gamma_fixed:
        return fmin_l_bfgs_b(
            func=negative_llf_separate_params,  # minimize this
            x0=np.array([n_st]),  # init. guess
            approx_grad=True,  # calculate gradient numerically
            args=(beta_start, gamma_start, observation, verbose),
            bounds=[n_bounds],
            iprint=1,
            # epsilon=step_size
        )
    else:
        return fmin_l_bfgs_b(
            func=negative_llf,  # minimize this
            x0=np.array([n_st, beta_start, gamma_start]),  # init. guess
            approx_grad=True,  # calculate gradient numerically
            args=(observation, verbose),
            bounds=[n_bounds, (eps, None), (eps, None)],
            iprint=1,
            # epsilon=step_size  # TODO: rm step_size arg
        )


def infection_times(observation):
    """
    Returns
    -------
    result : list
        Each list element gives the infection times of the corresponding
        simulation in self.data_. The infection times are represented by a
        1-dimensional array. The first self.i_0 entries are 0 because those
        infections occurred at time 0 or earlier.

    Examples
    --------
    >>> import pandas as pd
    >>> observation_df = pd.DataFrame(
    ...    {"t": [0.,1.,2.,4.,11],  # last event will be neglected because
    ...     "i": [5, 6, 6, 7, 8],   # t > t_max.
    ...     "s": [5, 4, 4, 3, 2],
    ...     "r": [0, 0, 1, 1, 1]}
    ... )
    >>> inf_times = infection_times(observation_df)
    >>> np.array_equal(inf_times, np.array([0, 0, 0, 0, 0, 1, 4]))
    True
    """
    i_0 = observation.i.iloc[0]
    return np.concatenate(
        (np.zeros(i_0),
         observation.t[observation.i.diff() == 1].values[:-1]
         )
    )


def transition_matrix(n, beta, gamma):
    """
    Parameters
    ----------
    n : int
        See the corresponding parameter in function :func:`_rates_sir`.
    beta : float
        See the corresponding parameter in function :func:`_rates_sir`.
    gamma : float
        See the corresponding parameter in function :func:`_rates_sir`.

    Returns
    -------
    matrix ; scipy.sparse.coo_matrix
        Sparse transition matrix where an entry at index (k, l) represents
        the probability of moving from state k to state l. The state
        (expressed as (number_of_susceptibles, number_of_infectetd)) that
        is referred to by k (or l) is int_pairs_leq_n_when_summed(n)[k]
        (or int_pairs_leq_n_when_summed(n)[l]).

    Examples
    --------
    >>> import numpy as np
    >>> obtained = transition_matrix(2, 1, .5).toarray()
    >>> desired = np.array([
    ... [ 1.,  1., 0., 0., 0.,  0.],
    ... [ 0.,  0., 1., 0., 0.,  0.],
    ... [ 0.,  0., 0., 0., 0.5, 0.],
    ... [ 0.,  0., 0., 1., 0.5, 0.],
    ... [ 0.,  0., 0., 0., 0.,  0.],
    ... [ 0.,  0., 0., 0., 0.,  1.]])
    >>> np.array_equal(obtained, desired)
    True
    """
    possible_states = int_pairs_leq_n_when_summed(n)
    matrix_size = len(possible_states)
    row = []  # row index of transition matrix
    col = []  # col index of transition matrix
    data = []  # entries of transition matrix
    for c, state in enumerate(possible_states):
        col.append(c)
        s, i = state
        # absorbing state
        if i == 0:
            row.append(c)  # we stay in the same state ...
            data.append(1)  # ... with probability 1.
        # non-absorbing state with no susceptible left
        elif s == 0:
            row.append(get_index(s, i-1, n))  # next event is a recovery...
            data.append(1)  # ... with probability 1
        # non-absorbing state
        else:
            col.append(c)  # 2nd col entry because we'll have 2 row entries
            # new infection
            row.append(get_index(s-1, i+1, n))
            prob_infect = (beta * s/n * i) / (beta * s/n * i + gamma * i)
            data.append(prob_infect)
            # new recovery
            row.append(get_index(s, i-1, n))
            data.append(1-prob_infect)
    trans_mat = coo_matrix((data, (row, col)),
                           shape=(matrix_size, matrix_size))
    return trans_mat.tocsr()


def _infection(state_old, state_new):
    """
    Parameters
    ----------
    state_old : dict or pd.Series
        Dictionary or pd.Series with the keys "s", "i", and "r".
    state_new : dict or pd.Series
        Same type requirements as for the `state_old` argument in this function
        apply.

    Returns
    -------
    infected : bool
        True if the event that occurred between `state_old` and `state_new` was
        an infection. False otherwise.

    """
    return state_new["s"] == state_old["s"] - 1 and \
        state_new["i"] == state_old["i"] + 1 and \
        state_new["r"] == state_old["r"]


def _recovery(state_old, state_new):
    """
    Parameters
    ----------
    state_old : dict or pd.Series
        Dictionary or pd.Series with the keys "s", "i", and "r".
    state_new : dict or pd.Series
        Same type requirements as for the `state_old` argument in this function
        apply.

    Returns
    -------
    recovered : bool
        True if the event that occurred between `state_old` and `state_new` was
        a recovery. False otherwise.
    """
    return state_new["s"] == state_old["s"] and \
        state_new["i"] == state_old["i"] - 1 and \
        state_new["r"] == state_old["r"] + 1

