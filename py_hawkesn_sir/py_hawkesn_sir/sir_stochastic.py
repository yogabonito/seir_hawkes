from math import log
from random import random, seed

from scipy.stats import expon
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .util import get_index, int_pairs_leq_n_when_summed


class StochasticSIR:
    def __init__(self, s_0, i_0, r_0=0, beta=0.1, gamma=0.2):
        """
        Class for simulating processes following a Continuous Time Markov Chain
        SIR model.
        
        Parameters
        ----------
        s_0 : int
            Number of susceptible individuals at time 0.
        i_0 : int
            Number of infected individuals at time 0.
        r_0 : int, default: 0
            Number of recovered individuals at time 0.
        beta : float, 0 <= beta <= 1, default: 0.1
            Infection rate. (If a susceptible and an infected individual meet,
            then the susceptible individual becomes infected at this rate.)
        gamma : float, 0 <= gamma <= 1, default: 0.2
            Recovery rate. (An infected individual recovers at this rate.)
        """
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.n = s_0 + i_0 + r_0
        self.beta = beta
        self.gamma = gamma
        self.birth_rate = 0  # TODO (if needed)
        self.death_rate = 0  # TODO (if needed)
        
    def simulate(self, t_max=10, n_simulations=1, random_state=None,
                 verbose=False):
        """
        Runs a simulation of the stochastic SIR model with the model's
        parameters provided as arguments. The result is a list in self.data_
        where each entry represents a simulation. Each simulation is a pandas
        `DataFrame` with the columns "s", "i", "r", and "t". The former three
        represent the number of individuals in the corresponding compartment
        (S, I, and R) and the latter represents the time of events (e.g. an
        infection).

        Parameters
        ----------
        t_max : float
            Only events in the time interval [0, t_max] are considered. Thus
            t_max must be >0.
        n_simulations : int, n_simulations > 0, default: 1
            Number of simulations to run.
        random_state : int or None
            Seed for random number generator. If None, the random number
            generator is not initialized and the current system time will be
            used.
        verbose : boolean, default: False
            If True, print output.
        """
        self.t_max_ = t_max
        self.n_simulations_ = n_simulations
        self.random_state_ = random_state
        seed(random_state)
        data = []
        for i in range(n_simulations):
            t = 0
            data.append([{"t": t,
                          "s": self.s_0,
                          "i": self.i_0,
                          "r": self.r_0}
            ])
            if verbose:
                print("Simulation starts with", data[i][0])

            while True:
                # compute rates
                state = data[i][-1]
                rate_i, rate_r, birth_death_rate_i, birth_rate_r, change_rate =\
                    self._rates_sir(state)

                next_state = state.copy()

                if change_rate > 0:
                    # compute time of next event
                    t += -log(random()) / change_rate
                    if t >= t_max:
                        break
                    # compute type of next event
                    unif_0_1 = random()
                    if unif_0_1 < rate_i / change_rate:
                        next_state["i"] += 1
                        next_state["s"] -= 1
                    elif unif_0_1 < (rate_i + rate_r) / change_rate:
                        next_state["i"] -= 1
                        next_state["r"] += 1
                    elif unif_0_1 < (rate_i + rate_r + birth_death_rate_i) / change_rate:
                        next_state["i"] -= 1
                        next_state["s"] += 1
                    else:
                        next_state["s"] += 1
                        self.n += 1  # not in the R code
                else:  # absorbing state reached ($I_t = 0$)
                    break

                next_state["t"] = t
                data[i].append(next_state)

                if verbose:
                    print("New state:", next_state)

            self.data_ = [pd.DataFrame(d) for d in data]
            if verbose:
                print("\nSimulation done!\n")

    def _rates_sir(self, state, beta=None, gamma=None, n=None):
        """
        Calculate the intensities of the cumulated infection process and the
        recovery process as well as birth/death rates.

        Parameters
        ----------
        state : dict or pd.Series
            Dictionary or pd.Series with the keys "s", "i", "r".
        beta : float or None, default: None
            If None, self.beta will be used, otherwise the provided beta will
            be used.
        gamma : float or None, default: None
            If None, self.gamma will be used, otherwise the provided gamma will
            be used.
        n : int or None, default: None
            If None, self.n will be used, otherwise the provided n will be
            used.

        Returns
        -------
        rate_i : float
            Rate at which an infection occurs.
        rate_r : float
            Rate at which a recovery occurs.
        birth_death_rate_i : float
            Rate at which a birth and death within the group of infected
            occurs. Note that this means one susceptible individual more and
            one infected individual less.
        birth_rate_r : float
            Rate at which a birth within the group of recovered occurs. Note
            that this means one susceptible individual more (the number of
            recovered individuals stays the same).
        change_rate : float
            Sum of the other returned rates.
        """
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        n = self.n if n is None else n
        rate_i = beta * (state["s"] / n) * state["i"]
        rate_r = gamma * state["i"]
        birth_death_rate_i = (self.birth_rate - self.death_rate) * state["i"]
        birth_rate_r = self.birth_rate * state["r"]
        change_rate = rate_i + rate_r + birth_death_rate_i + birth_rate_r
        return rate_i, rate_r, birth_death_rate_i, birth_rate_r, change_rate

    def plot(self, width=5.51, height=4, n_xticks=6, fname=None):
        """
        Plot the simulations made with .

        Parameters
        ----------
        width : float, default: 5.51
            Width of the plot.
        height : float, default: 4
            Height of the plot.
        n_xticks : int (must be non-negative), default: 6
            Number of ticks on the x-axis.
        fname : str, default: None
            Name of the file the plot is saved to. If `None`, the plot is not
            saved.
        """
        plt.figure(dpi=300, figsize=(width, height))
        alpha = min(1, max(0.01, 50/self.n_simulations_))
        for i in range(self.n_simulations_):
            obs = self.data_[i]
            last_state = obs.iloc[-1]
            last_state.t = self.t_max_
            obs = obs.append([last_state], ignore_index=True)
            plt.plot(obs[["t"]], obs[["s"]], c="C0", alpha=alpha)  # S
            plt.plot(obs[["t"]], obs[["i"]], c="C1", alpha=alpha)  # I
            plt.plot(obs[["t"]], obs[["r"]], c="C2", alpha=alpha)  # R
        title = "Stochastic SIR model "
        if self.n_simulations_ > 1:
            title += "(" + str(self.n_simulations_) + " samples) "
        title += "with " \
                 "$\\beta=" + str(self.beta) + "$ and " \
                 "$\\gamma=" + str(self.gamma) + "$"
        plt.title(title)
        plt.xlim([0, self.t_max_])
        plt.ylim([0, self.n])
        plt.xticks(np.linspace(0, self.t_max_, n_xticks))
        plt.xlabel("$t$")
        plt.ylabel("$S_t,\ I_t,\ R_t$")
        plt.grid()
        legend = plt.legend(["$S_t$", "$I_t$", "$R_t$"])
        for l in legend.get_lines():
            l.set_alpha(1)
        if fname is not None:
            plt.savefig(fname + ".pdf")

    def llf(self, n, beta, gamma, simulation_index):
        """
        Compute the log-likelihood of the simulation specified by
        `simulation_index` given the parameters N, :math:`\\beta`, and
        :math:`\\gamma`.

        Parameters
        ----------
        n : float
            Total number of individuals in the SIR model.
        beta : float
            Parameter beta in the SIR model. See corresponding argument in
            :meth:`simulate` for more information.
        gamma : float
            Parameter gamma in the SIR model. See corresponding argument in
            :meth:`simulate` for more information.
        simulation_index : int, 0 <= `simulation_number` < `len(self.data_)`
            Index specifying the simulation in `self.data_`.

        Returns
        -------
        llf : float
            The log-likelihood.

        Notes
        -----
        The main difference to the API of the R code is the missing I.0 parameter
        which is not used in the R code as well.
        """
        simulation = self.data_[simulation_index]
        llf = 0
        state_new = simulation.iloc[0]
        t_new = state_new["t"]
        for j in range(1, simulation.shape[0]):
            t_old = t_new
            state_old = state_new
            state_new = dict(simulation.iloc[j])
            t_new = state_new["t"]
            rate_i, rate_r, birth_death_rate_i, birth_rate_r, change_rate = \
                self._rates_sir(state_old, beta, gamma, n)
            if change_rate == 0:
                break
            # likelihood of waiting the time we waited
            # NOTE: R code uses `rate=change_rate` while in Python it is
            # `scale=1/change_rate`
            llf += expon.logpdf(t_new - t_old, scale=1 / change_rate)
            # likelihood of observing the event we observed
            if _infection(state_old, state_new):
                llf += log(rate_i / change_rate)
            elif _recovery(state_old, state_new):
                llf += log(rate_r / change_rate)
            elif _death_birth_i(state_old, state_new):
                llf += log(birth_death_rate_i / change_rate)
            elif _birth_r(state_old, state_new):
                llf += log(birth_rate_r / change_rate)
                n += 1

        return llf

    def fit(self, n_start=None, beta_start=0.1, gamma_start=0.2,
            verbose=False):
        """
        Parameters
        ----------
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

        Returns
        -------
        results : list
            Each list entry contains the result of fitting one
            simulation. Each of these results is produced by the function
            :func:`fmin_l_bfgs_b` which implements the L-BFGS algorithm (see
            [1]_).
            In addition to returning the estimation results the results are
            also to the `est_n_beta_gamma_` attribute of the object.

        Notes
        -----
        In contrast to the R code the parameter of initially infected
        individuals (I.0 in the R code) is not used.

        References
        ----------
        This method uses the L-BFGS algorithm (see [1]_).
        .. [1] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization (1997), ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
        """
        def negative_llf(n_beta_gamma, simulation_index):
            """
            Parameters
            ----------
            n_beta_gamma : np.array (shape (3))
                Values for N, beta, and gamma provided in a single array.
            simulation_index : int
                Index specifying the simulation in self.data_.

            Returns
            -------
            neg_llf : float
                The negative log-likelihood.
            """
            n, beta, gamma = tuple(n_beta_gamma)
            if verbose:
                print("  args n, beta, gamma:", n, beta, gamma)
            return -self.llf(n, beta, gamma, simulation_index)

        eps = np.finfo(float).eps
        results = [None] * self.n_simulations_

        for obs_index in range(self.n_simulations_):
            if n_start is None:
                sim = self.data_[obs_index]
                n_st = sim.i.iloc[-1] + sim.r.iloc[-1]
                n_bounds = (n_st, None)
            else:
                n_st = n_start
                n_bounds = (0, None)

            if verbose:
                print("Fit observation number", obs_index)
                print("n_start is", n_start)

            results[obs_index] = fmin_l_bfgs_b(
                func=negative_llf,  # minimize this
                x0=np.array([n_st, beta_start, gamma_start]), # init. guess
                approx_grad=True,  # calculate gradient numerically
                args=(obs_index,),
                bounds=[n_bounds, (eps, None), (eps, None)],
                iprint=1
            )

        self.est_n_beta_gamma_ = results
        return self.est_n_beta_gamma_

    def infection_times(self, until=None):
        """
        Parameters
        ----------
        until : float or None, default: None
            Only infections until the point in time specified by `until` will
            be considered.

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
        >>> example_df = pd.DataFrame(
        ...    {"t": [0.,1.,2.,4.,11],  # last event will be neglected because
        ...     "i": [5, 6, 6, 7, 8],   # t > t_max.
        ...     "s": [5, 4, 4, 3, 2],
        ...     "r": [0, 0, 1, 1, 1]}
        ... )
        >>> example_sir = StochasticSIR(s_0=5, i_0=5)
        >>> example_sir.data_ = [example_df, example_df]  # two "simulations"
        >>> example_sir.n_simulations_ = len(example_sir.data_)
        >>> inf_times = example_sir.infection_times()
        >>> # first simulation
        >>> np.array_equal(inf_times[0], np.array([0, 0, 0, 0, 0, 1, 4]))
        True
        >>> # second simulation
        >>> np.array_equal(inf_times[1], np.array([0, 0, 0, 0, 0, 1, 4]))
        True
        >>> # and now the same with the `until` argument
        >>> inf_times = example_sir.infection_times(until=3)
        >>> np.array_equal(inf_times[0], np.array([0, 0, 0, 0, 0, 1]))
        True
        >>> np.array_equal(inf_times[1], np.array([0, 0, 0, 0, 0, 1]))
        True
        """
        inf_times_list = [np.concatenate(
            (np.zeros(self.i_0),
             self.data_[sim]["t"][self.data_[sim]["i"].diff()==1].values[:-1]))
            for sim in range(self.n_simulations_)
        ]
        if until is None:
            return inf_times_list
        else:
            return [inf_times[inf_times <= until]
                    for inf_times in inf_times_list]

    @staticmethod
    def transition_matrix(n, beta, gamma):
        """
        Parameters
        ----------
        n : int
            Total population.
        beta : float
            See :meth:`__init__` for more information.
        gamma : float
            See :meth:`__init__` for more information.

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
        >>> obtained = StochasticSIR.transition_matrix(2, 1, .5).toarray()
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
        transition_matrix = coo_matrix((data, (row, col)),
                                       shape=(matrix_size, matrix_size))
        return transition_matrix.tocsr()


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


def _death_birth_i(state_old, state_new):
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
    death_birth : bool
        True if the event that occurred between `state_old` and `state_new` was
        a death of an infected individual and a birth of a new susceptible
        individual. False otherwise.
    """
    return state_new["s"] == state_old["s"] + 1 and \
        state_new["i"] == state_old["i"] - 1 and \
        state_new["r"] == state_old["r"]


def _birth_r(state_old, state_new):
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
    birth : bool
        True if the event that occurred between `state_old` and `state_new` was
        a birth of a new susceptible individual. False otherwise.
    """
    return state_new["s"] == state_old["s"] + 1 and \
        state_new["i"] == state_old["i"] and \
        state_new["r"] == state_old["r"]
