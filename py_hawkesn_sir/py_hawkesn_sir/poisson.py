from random import random, seed

import numpy as np
import matplotlib.pyplot as plt


class Poisson:
    """
    Class for simulating Poisson processes.
    """
    def __init__(self, intensity):
        self.intensity = intensity

    def simulate(self, t_max=10, n_simulations=1, random_state=None,
                 verbose=False):
        self.t_max_ = t_max
        self.n_simulations_ = n_simulations
        self.random_state_ = random_state
        np.random.seed(random_state)
        self.data_ = []
        for i in range(n_simulations):
            waiting_times = np.random.exponential(
                scale=1 / self.intensity, size=100)
            jump_times = np.cumsum(waiting_times)
            while jump_times[-1] < t_max:
                new_waiting_times = np.random.exponential(
                    scale=1 / self.intensity, size=100)
                new_jump_times = np.cumsum(new_waiting_times) + jump_times[-1]
                jump_times = np.concatenate((jump_times, new_jump_times))
            n_jumps_before_t_max = sum(jump_times < t_max)
            jump_times_head = jump_times[:n_jumps_before_t_max + 1]
            jump_times_and_zero = np.concatenate(([0], jump_times_head))
            self.data_.append(jump_times_and_zero)

        if verbose:
            print("\nSimulation done!\n")

    def plot(self, width=5.51, height=4, n_xticks=6, fname=None, simulation=0):
        """
        Plot one simulated path.

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
        simulation : int (must be in [0, self.n_simulations_]), default: 0
            Index of the simulation to plot.
        """
        plt.figure(dpi=300, figsize=(width, height))
        jump_times_and_zero = self.data_[simulation]
        jump_times = jump_times_and_zero[1:]
        # lines in the plot
        plt.hlines(np.arange(len(jump_times)),
                   jump_times_and_zero[:-1],
                   jump_times,
                   color="C0")
        # dots in the plot
        plt.plot(jump_times_and_zero,
                 np.arange(len(jump_times) + 1),
                 linestyle='None',
                 marker="o")

        plt.title("Poisson process with intensity " + str(self.intensity))
        plt.xlim(0, self.t_max_)
        plt.xticks(np.linspace(0, self.t_max_, n_xticks))
        plt.xlabel("$t$")
        plt.ylabel("$N_t$")
        if fname is not None:
            plt.savefig(fname + ".pdf")
