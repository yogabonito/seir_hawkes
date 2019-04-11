from math import log
from random import random, seed

from scipy.stats import expon
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

from .util import get_index, int_pairs_leq_n_when_summed


class StochasticSEIR:
    def __init__(self, s_0, e_0, i_0, r_0=0, beta=0.2, sigma=0.2, gamma=0.2):
        """
        Class for simulating processes following a Continuous Time Markov Chain
        SEIR model.

        Parameters
        ----------
        s_0 : int
            Number of susceptible individuals at time 0.
        e_0 : int
            Number of exposed (infected but not infectious) individuals at time
            0.
        i_0 : int
            Number of infected and infectious individuals at time 0.
        r_0 : int, default: 0
            Number of recovered individuals at time 0.
        beta : float, 0 <= beta <= 1, default: 0.1
            Infection rate. (If a susceptible and an infected individual meet,
            then the susceptible individual becomes exposed at this rate. In
            other words, rate of transition S --> E.)
        sigma : float, 0 <= beta <= 1, default: 0.1
            An infection's activation rate. (Rate of transition E --> I.)
        gamma : float, 0 <= gamma <= 1, default: 0.2
            Recovery rate. (Rate of transition I --> R.)
        """
        self.s_0 = s_0
        self.e_0 = e_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.n = s_0 + e_0 + i_0 + r_0
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

    def simulate(self, t_max=10, n_simulations=1, random_state=None,
                 verbose=False):
        """
        Runs a simulation of the stochastic SEIR model with the model's
        parameters provided as arguments. The result is a list in self.data_
        where each entry represents a simulation. Each simulation is a pandas
        `DataFrame` with the columns "s", "e", "i", "r", and "t". The former
        four represent the number of individuals in the corresponding
        compartment (S, E, I, and R) and the latter represents the time of
        events (e.g. a recovery).

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
                          "e": self.e_0,
                          "i": self.i_0,
                          "r": self.r_0}
                         ])
            if verbose:
                print("Simulation starts with", data[i][0])

            while True:
                # compute rates
                state = data[i][-1]
                rate_e, rate_i, rate_r, change_rate = self._rates_sir(state)

                next_state = state.copy()

                if change_rate > 0:
                    # compute time of next event
                    t += -log(random()) / change_rate
                    if t >= t_max:
                        break
                    # compute type of next event
                    unif_0_1 = random()
                    if unif_0_1 < rate_e / change_rate:
                        next_state["s"] -= 1
                        next_state["e"] += 1
                    elif unif_0_1 < (rate_e + rate_i) / change_rate:
                        next_state["e"] -= 1
                        next_state["i"] += 1
                    elif unif_0_1 < (rate_e + rate_i + rate_r) / change_rate:
                        next_state["i"] -= 1
                        next_state["r"] += 1
                else:  # absorbing state reached ($E_t = I_t = 0$)
                    break

                next_state["t"] = t
                data[i].append(next_state)

                if verbose:
                    print("New state:", next_state)

            self.data_ = [pd.DataFrame(d) for d in data]
            if verbose:
                print("\nSimulation done!\n")

    def _rates_sir(self, state, beta=None, sigma=None, gamma=None, n=None):
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
        sigma : float or None, default: None
            If None, self.sigma will be used, otherwise the provided sigma will
            be used.
        gamma : float or None, default: None
            If None, self.gamma will be used, otherwise the provided gamma will
            be used.
        n : int or None, default: None
            If None, self.n will be used, otherwise the provided n will be
            used.

        Returns
        -------
        rate_e : float
            Rate at which an infection occurs (S --> E).
        rate_i : float
            Rate at which an infection becomes active (E --> I).
        rate_r : float
            Rate at which a recovery occurs.
        change_rate : float
            Sum of the other returned rates.
        """
        beta = self.beta if beta is None else beta
        sigma = self.sigma if sigma is None else sigma
        gamma = self.gamma if gamma is None else gamma
        n = self.n if n is None else n
        rate_e = beta * (state["s"] / n) * state["i"]
        rate_i = sigma * state["e"]
        rate_r = gamma * state["i"]
        change_rate = rate_e + rate_i + rate_r
        return rate_e, rate_i, rate_r, change_rate

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
            plt.plot(obs[["t"]], obs[["e"]], c="C3", alpha=alpha)  # E
            plt.plot(obs[["t"]], obs[["i"]], c="C1", alpha=alpha)  # I
            plt.plot(obs[["t"]], obs[["r"]], c="C2", alpha=alpha)  # R
        title = "Stochastic SEIR model "
        if self.n_simulations_ > 1:
            title += "(" + str(self.n_simulations_) + " samples) "
        title += "with " \
                 "$\\beta=" + str(self.beta) + "$, " \
                 "$\\sigma=" + str(self.sigma) + "$, and " \
                 "$\\gamma=" + str(self.gamma) + "$"
        plt.title(title)
        plt.xlim([0, self.t_max_])
        plt.ylim([0, self.n])
        plt.xticks(np.linspace(0, self.t_max_, n_xticks))
        plt.xlabel("$t$")
        plt.ylabel("$S_t,\ E_t,\ I_t,\ R_t$")
        plt.grid()
        legend = plt.legend(["$S_t$", "$E_t$", "$I_t$", "$R_t$"])
        for l in legend.get_lines():
            l.set_alpha(1)
        if fname is not None:
            plt.savefig(fname + ".pdf")

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
            1-dimensional array. The first self.e_0 entries are 0 because those
            infections occurred at time 0 or earlier.

        Examples
        --------
        >>> import pandas as pd
        >>> example_df = pd.DataFrame(
        ...    {"t": [0.,1.,2.,4.,11, 12],  # last event will be neglected because
        ...     "e": [5, 6, 5, 5, 6, 5],
        ...     "i": [5, 5, 6, 5, 5, 6],   # t > t_max.
        ...     "s": [5, 4, 4, 4, 3, 3],
        ...     "r": [0, 0, 0, 1, 1, 1]}
        ... )
        >>> example_seir = StochasticSEIR(s_0=5, e_0=5, i_0=5)
        >>> example_seir.data_ = [example_df, example_df]  # two "simulations"
        >>> example_seir.n_simulations_ = len(example_seir.data_)
        >>> inf_times = example_seir.infection_times()
        >>> # first simulation
        >>> np.array_equal(inf_times[0], np.array([0, 0, 0, 0, 0, 1, 11]))
        True
        >>> # second simulation
        >>> np.array_equal(inf_times[1], np.array([0, 0, 0, 0, 0, 1, 11]))
        True
        >>> # and now the same with the `until` argument
        >>> inf_times = example_seir.infection_times(until=3)
        >>> np.array_equal(inf_times[0], np.array([0, 0, 0, 0, 0, 1]))
        True
        >>> np.array_equal(inf_times[1], np.array([0, 0, 0, 0, 0, 1]))
        True
        """
        inf_times_list = [np.concatenate(
            (np.zeros(self.e_0),
             self.data_[sim]["t"][self.data_[sim]["e"].diff()==1].values))#[:-1]))
            for sim in range(self.n_simulations_)
        ]
        if until is None:
            return inf_times_list
        else:
            return [inf_times[inf_times <= until]
                    for inf_times in inf_times_list]
