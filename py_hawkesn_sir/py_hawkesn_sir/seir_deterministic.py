import numpy as np
import matplotlib.pyplot as plt


class DeterministicSEIR:
    def __init__(self, beta, sigma, gamma, s_0, e_0, i_0, r_0=0):
        """
        Parameters
        ----------
        beta : float, >=0
            Infection rate. (If a susceptible and an infected individual meet,
            then the susceptible individual becomes exposed at this rate. In
            other words, rate of transition S --> E.)
        sigma :  float, >=0
            An infection's activation rate. (Rate of transition E --> I.)
        gamma : float, >=0
            Recovery rate. (Rate of transition I --> R.)
        s_0 : int
            Number of susceptible individuals at the beginning.
        e_0 : int
            Number of exposed individuals at the beginning.
        i_0 : int
            Number of infected individuals at the beginning.
        r_0 : int, default: 0
            Number of recovered individuals at the beginning.
        """
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.s_0 = s_0
        self.e_0 = e_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.n = s_0 + e_0 + i_0 + r_0
        self.n_compartments = 4

    def solve_euler(self, t_end=10, n_steps=100):
        """
        Calculates S(t), E(t), I(t), and R(t) for :math:`t \in [0, t_end]`
        according to the Euler method. The interval
        :math:`[0, t_end]` is partitioned into `n_steps` intervals.

        Parameters
        ----------
        t_end : int
            Time until which we want to calculate S(t), E(t), I(t), and
            R(t).
        n_steps : int
            Number of time steps to calculate.
        """
        s_, e_, i_, r_ = [np.empty(n_steps + 1)
                          for _ in range(self.n_compartments)]
        s_[0], e_[0], i_[0], r_[0] = self.s_0, self.e_0, self.i_0, self.r_0

        delta_t = t_end / n_steps
        for t in range(1, n_steps + 1):
            s_lost = delta_t * self.beta * s_[t - 1] * i_[t - 1] / self.n
            e_lost = delta_t * self.sigma * e_[t - 1]
            i_lost = delta_t * self.gamma * i_[t - 1]
            s_[t] = s_[t - 1] - s_lost
            e_[t] = e_[t - 1] + s_lost - e_lost
            i_[t] = i_[t - 1] + e_lost - i_lost
            r_[t] = r_[t - 1] + i_lost
        self.s_, self.e_, self.i_, self.r_ = s_, e_, i_, r_
        self.t_end = t_end
        self.n_steps = n_steps
        return self

    def plot(self, width=5.51, height=4, n_xticks=6, fname="seir_det"):
        t = np.linspace(0, self.t_end, self.n_steps + 1)
        plt.figure(dpi=300, figsize=(width, height))
        plt.title("Deterministic SEIR model with " +
                  "$\\beta=" + str(self.beta) + "$,  " +
                  "$\\sigma=" + str(self.sigma) + "$, and " +
                  "$\\gamma=" + str(self.gamma) + "$")
        plt.xticks(np.linspace(0, self.t_end, n_xticks))
        plt.xlabel("$t$")
        plt.ylabel("$S(t),\ E(t),\ I(t),\ R(t)$")
        plt.plot(t, self.s_, label="$S(t)$", c="C0")
        plt.plot(t, self.e_, label="$E(t)$", c="C3")
        plt.plot(t, self.i_, label="$I(t)$", c="C1")
        plt.plot(t, self.r_, label="$R(t)$", c="C2")

        plt.xlim([0, self.t_end])
        plt.ylim([0, self.n])
        plt.grid()
        plt.legend()

        plt.savefig(fname + ".pdf")
