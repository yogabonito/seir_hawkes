import numpy as np
import matplotlib.pyplot as plt


class DeterministicSIR:
    def __init__(self, beta, gamma, s_0, i_0, r_0=0):
        """
        Parameters
        ----------
        beta : float, >=0
            Infection rate.
        gamma : float, >=0
            Recovery rate.
        s_0 : int
            Number of susceptible individuals at the beginning.
        i_0 : int
            Number of infected individuals at the beginning.
        r_0 : int, default: 0
            Number of recovered individuals at the beginning.
        """
        self.beta = beta
        self.gamma = gamma
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.n = s_0 + i_0 + r_0

    def solve_euler(self, t_end=10, n_steps=100):
        """
        Calculates S(t), I(t), and R(t) for :math:`t \in [0, t_end]`
        according to the Euler method. The intervall
        :math:`[0, t_end]` is partitioned into `n_steps` intervals.

        Parameters
        ----------
        t_end : int
            Time until which we want to calculate S(t), I(t), and
            R(t).
        n_steps : int
            Number of time steps to calculate.
        """
        s_, i_, r_ = [np.empty(n_steps + 1) for _ in range(3)]
        s_[0], i_[0], r_[0] = self.s_0, self.i_0, self.r_0

        delta_t = t_end / n_steps
        for t in range(1, n_steps + 1):
            s_[t] = s_[t - 1] - delta_t * \
                self.beta * s_[t - 1] * i_[t - 1] / self.n
            i_[t] = i_[t - 1] + delta_t * i_[t - 1] * \
                (self.beta * s_[t - 1] / self.n - self.gamma)
            r_[t] = r_[t - 1] + delta_t * i_[t - 1] * self.gamma
        self.s_, self.i_, self.r_ = s_, i_, r_
        self.t_end = t_end
        self.n_steps = n_steps
        return self

    def plot(self, width=5.51, height=4, n_xticks=6, fname=None):
        t = np.linspace(0, self.t_end, self.n_steps + 1)
        plt.figure(dpi=300, figsize=(width, height))
        plt.title("Deterministic SIR model with " +
                  "$\\beta=" + str(self.beta) + "$ and " +
                  "$\\gamma=" + str(self.gamma) + "$")
        plt.xticks(np.linspace(0, self.t_end, n_xticks))
        plt.xlabel("$t$")
        plt.ylabel("$S(t),\ I(t),\ R(t)$")
        plt.plot(t, self.s_, label="$S(t)$")
        plt.plot(t, self.i_, label="$I(t)$")
        plt.plot(t, self.r_, label="$R(t)$")

        plt.xlim([0, self.t_end])
        plt.ylim([0, self.n])
        plt.grid()
        plt.legend()
        
        if fname:
            plt.savefig(fname + ".pdf")

