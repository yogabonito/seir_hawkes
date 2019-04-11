from math import log

from scipy.stats import expon
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd

from .util import get_index_for_triples, int_triples_leq_n_when_summed


def _rates_seir(state, beta, sigma, gamma, n):
    """
    Calculate the intensities of the cumulated infection process and the
    recovery process.

    Parameters
    ----------
    state : dict or pd.Series
        Dictionary or pd.Series with the keys "s", "e", "i", "r". The values
        associated with the keys are integers.
    beta : float
        Parameter beta of the SEIR model.
    sigma : float
        Parameter sigma of the SEIR model.
    gamma : float
        Parameter gamma of the SEIR model.
    n : int
        Parameter N of the SEIR model.

    Returns
    -------
    rate_e : float
        Rate at which a susceptible individual becomes exposed.
    rate_i : float
        Rate at which an exposed individual becomes infectious.
    rate_r : float
        Rate at which a recovery occurs.
    change_rate : float
        Sum of the other returned rates.
    """
    rate_e = beta * (state["s"] / n) * state["i"]
    rate_i = sigma * state["e"]
    rate_r = gamma * state["i"]
    change_rate = rate_e + rate_i + rate_r
    return rate_e, rate_i, rate_r, change_rate


def llf(observation, n, beta, sigma, gamma, verbose):
    """
    Compute the log-likelihood of the simulation specified by
    `simulation_index` given the parameters N, :math:`\\beta`, and
    :math:`\\gamma`.

    Parameters
    ----------
    observation : pandas.DataFrame
        DataFrame representing the observed SEIR model. Columns are "s", "i",
        "r", and "t" for the compartments S, I, and R and the time,
        respectively.
    n : int or float
        Total number of individuals in the SEIR model.
    beta : float
        Parameter beta in the SEIR model. See corresponding argument in
        :meth:`simulate` for more information.
    sigma : float
        Parameter sigma in the SEIR model. See corresponding argument in
        :meth:`simulate` for more information.
    gamma : float
        Parameter gamma in the SEIR model. See corresponding argument in
        :meth:`simulate` for more information.
    verbose : bool
        If True, print debug output.

    Returns
    -------
    llf : float
        The log-likelihood.
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
        rate_e, rate_i, rate_r, change_rate = _rates_seir(state_old, beta,
                                                          sigma, gamma, n)
        if change_rate == 0:
            break
        # expon.pdf(t_new - t_old, scale=1 / change_rate) gives the likelihood
        # of waiting the time we waited
        # NOTE: R code uses `rate=change_rate` while in Python it is
        # `scale=1/change_rate`
        llf += expon.logpdf(t_new - t_old, scale=1 / change_rate)
        # likelihood of observing the event we observed
        if _infection_latent(state_old, state_new):
            llf += log(rate_e / change_rate)
        elif _infection_active(state_old, state_new):
            llf += log(rate_i / change_rate)
        elif _recovery(state_old, state_new):
            llf += log(rate_r / change_rate)
    return llf


def negative_llf(n_beta_sigma_gamma, observation, verbose):
    """
    Wrapper for :func:`llf` taking n, beta, gamma as a single array and
    returning the *negative* log-likelihood.

    Parameters
    ----------
    n_beta_sigma_gamma : np.array (shape (3))
        Values for N, beta, sigma, and gamma provided in a single array.
    observation : pandas.DataFrame
        See corresponding parameter in :func:`llf`.
    verbose : bool
        If True, print arguments.

    Returns
    -------
    neg_llf : float
        The negative log-likelihood.
    """
    n, beta, sigma, gamma = tuple(n_beta_sigma_gamma)
    if verbose:
        print("  args n, beta, sigma, gamma:", n, beta, sigma, gamma)
    return -llf(observation, n, beta, sigma, gamma, verbose)


def negative_llf_separate_params(n, beta, sigma, gamma, observation, verbose):
    """
    Same as :func:`negative_llf`. The only difference is that the four
    parameters n, beta, sigma, and gamma are provided as separate arguments and
    not in a single array. This makes the function suitable for optimization
    purposes where beta, sigma, and gamma are fixed.
    """
    if verbose:
        print("  args n, beta, sigma, gamma:", n, beta, sigma, gamma)
    return -llf(observation, n, beta, sigma, gamma, verbose)


def fit(observation, n_start=None, beta_start=0.2, sigma_start=0.2,
        gamma_start=0.2, verbose=False, observation_id=None,
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
    sigma_start : float
        Starting value for the likelihood optimization.
    gamma_start : float
        Starting value for the likelihood optimization.
    verbose : bool, default: False
        If True, output is printed during the fitting.
    observation_id : int or None
        If int and verbose==True, then display the id of the observation.
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
        n_st = observation.e.iloc[-1] + observation.i.iloc[-1] \
               + observation.r.iloc[-1]
        n_bounds = (n_st, None)
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
            x0=n_st,  # init. guess
            approx_grad=True,  # calculate gradient numerically
            args=(beta_start, sigma_start, gamma_start, observation, verbose),
            bounds=[n_bounds],
            iprint=1,
        )
    else:
        return fmin_l_bfgs_b(
            func=negative_llf,  # minimize this
            x0=np.array([n_st, beta_start, sigma_start,
                         gamma_start]),  # init. guess
            approx_grad=True,  # calculate gradient numerically
            args=(observation, verbose),
            bounds=[n_bounds, (eps, None), (eps, None), (eps, None)],
            iprint=1,
        )


def infection_times(observation, until=None, infection="e"):
    """
    Parameters
    ----------
    observation : pd.DataFrame
        DataFrame containing the columns "s", "e", "i", "r", and "t".
    until : float or None, default: None
        Only infections until the point in time specified by `until` will
        be considered.
    infection : str, {"e", "i"}. default: "e"
        Specifies what is considered an infection. If "e" ("i"), an infection
        is the transition from S to E (E to I).

    Returns
    -------
    result : list
        Each list element gives the infection times of the corresponding
        simulation in observation. The infection times are represented by a
        1-dimensional array. The first entries are 0 because those infections
        occurred at time 0 or earlier.

    Examples
    --------
    >>> import pandas as pd
    >>> observation_df = pd.DataFrame(
    ...    {"t": [0.,1.,2.,4.,11, 12, 13, 14],
    ...     "e": [5, 6, 5, 4, 4, 5, 4, 4],
    ...     "i": [2, 2, 3, 4, 3, 3, 4, 3],
    ...     "s": [5, 4, 4, 4, 4, 3, 3, 3],
    ...     "r": [0, 0, 0, 0, 1, 1, 1, 2]}
    ... )
    >>> inf_times = infection_times(observation_df)
    >>> np.array_equal(inf_times, np.array([0, 0, 0, 0, 0, 1., 12.]))
    True
    >>> inf_times = infection_times(observation_df, until=12)
    >>> np.array_equal(inf_times, np.array([0, 0, 0, 0, 0, 1., 12.]))
    True
    >>> inf_times = infection_times(observation_df, until=11.9)
    >>> np.array_equal(inf_times, np.array([0, 0, 0, 0, 0, 1.]))
    True
    >>> inf_times = infection_times(observation_df, until=1.0)
    >>> np.array_equal(inf_times, np.array([0, 0, 0, 0, 0, 1.]))
    True
    >>> inf_times = infection_times(observation_df, until=.9)
    >>> np.array_equal(inf_times, np.array([0, 0, 0, 0, 0]))
    True
    >>> inf_times = infection_times(observation_df, infection="i")
    >>> np.array_equal(inf_times, np.array([0, 0, 2., 4., 13.]))
    True
    >>> inf_times = infection_times(observation_df, infection="i", until=12)
    >>> np.array_equal(inf_times, np.array([0, 0, 2., 4.]))
    True
    """
    if infection == "e":
        infections_at_start = observation.e.iloc[0]
    elif infection == "i":
        infections_at_start = observation.i.iloc[0]
    else:
        raise ValueError("The infection argument must be either 'e' or 'i'.")

    inf_times = np.concatenate(
        (np.zeros(infections_at_start),
         observation.t[observation[infection].diff() == 1].values
         )
    )

    if until is None:
        return inf_times
    else:
        return inf_times[inf_times <= until]


def transition_matrix(n, beta, sigma, gamma):
    """
    Parameters
    ----------
    n : int
        See the corresponding parameter in function :func:`_rates_seir`.
    beta : float
        See the corresponding parameter in function :func:`_rates_seir`.
    sigma : float
        See the corresponding parameter in function :func:`_rates_seir`.
    gamma : float
        See the corresponding parameter in function :func:`_rates_seir`.

    Returns
    -------
    matrix ; scipy.sparse.coo_matrix
        Sparse transition matrix where an entry at index (k, l) represents
        the probability of moving from state k to state l. The state -
        expressed as
        (number_of_susceptibles, number_of_exposed, number_of_infectetd) - that
        is referred to by k (or l) is int_triples_leq_n_when_summed(n)[k]
        (or int_triples_leq_n_when_summed(n)[l]).

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
    possible_states = int_triples_leq_n_when_summed(n)
    matrix_size = len(possible_states)
    row = []  # row index of transition matrix
    col = []  # col index of transition matrix
    data = []  # entries of transition matrix
    for c, state in enumerate(possible_states):
        col.append(c)
        s, e, i = state
        # absorbing state
        if e == 0 and i == 0:
            row.append(c)  # we stay in the same state ...
            data.append(1)  # ... with probability 1.
        # non-absorbing state
        elif i == 0:  # e > 0
            # E-->I
            row.append(get_index_for_triples(s, e-1, i+1, n))  # next event: activation...
            data.append(1)  # ... with probability 1
        elif e == 0:  # i > 0
            # S-->E or I-->R
            if s == 0:  # only I-->R
                row.append(get_index_for_triples(s, e, i - 1, n))
                data.append(1)
            else:
                col.append(c)  # 2nd col entry because we'll have 2 row entries
                # new infection
                row.append(get_index_for_triples(s-1, e+1, i, n))
                prob_infect = (beta * s/n * i) / (beta * s/n * i + gamma * i)
                data.append(prob_infect)
                # new recovery
                row.append(get_index_for_triples(s, e, i-1, n))
                data.append(1 - prob_infect)
        else:
            # S-->E or E-->I or I-->R
            if s == 0:  # only E-->I or I-->R
                col.append(c)  # 2nd col entry because we'll have 2 row entries
                # new activation
                row.append(get_index_for_triples(s, e-1, i+1, n))
                prob_act = sigma * e / (sigma * e + gamma * i)
                data.append(prob_act)
                # new recovery
                row.append(get_index_for_triples(s, e, i-1, n))
                data.append(1 - prob_act)
            else:
                col.append(c)  # 2nd col entry ...
                col.append(c)  # and 3rd because we'll have 3 row entries
                denominator = (beta * s/n * i + sigma * e + gamma * i)
                # new infection
                row.append(get_index_for_triples(s-1, e+1, i, n))
                prob_infect = (beta * s/n * i) / denominator
                data.append(prob_infect)
                # new activation
                row.append(get_index_for_triples(s, e-1, i+1, n))
                prob_act = sigma * e / denominator
                data.append(prob_act)
                # new recovery
                row.append(get_index_for_triples(s, e, i-1, n))
                data.append(1 - prob_infect - prob_act)
    trans_mat = coo_matrix((data, (row, col)),
                           shape=(matrix_size, matrix_size))
    return trans_mat.tocsr()


def _infection_latent(state_old, state_new):
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
    infection_latent : bool
        True if the event that occurred between `state_old` and `state_new` was
        a transition from S to E. False otherwise.

    """
    return state_new["s"] == state_old["s"] - 1 and \
        state_new["e"] == state_old["e"] + 1 and \
        state_new["i"] == state_old["i"] and \
        state_new["r"] == state_old["r"]


def _infection_active(state_old, state_new):
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
    infection_active : bool
        True if the event that occurred between `state_old` and `state_new` was
        a transition from E to I. False otherwise.

    """
    return state_new["s"] == state_old["s"] and \
        state_new["e"] == state_old["e"] - 1 and \
        state_new["i"] == state_old["i"] + 1 and \
        state_new["r"] == state_old["r"]


def _recovery(state_old, state_new):
    """
    Parameters
    ----------
    state_old : dict or pd.Series
        Dictionary or pd.Series with the keys "s", "e", "i", and "r".
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
        state_new["e"] == state_old["e"] and \
        state_new["i"] == state_old["i"] - 1 and \
        state_new["r"] == state_old["r"] + 1

