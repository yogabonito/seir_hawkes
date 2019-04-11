from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import matplotlib.pyplot as plt
from sympy import derive_by_array, exp, lambdify, log, Piecewise, symbols

from .sir_stochastic import StochasticSIR


class HawkesN:
    def __init__(self, history):
        """

        Parameters
        ----------
        history : sympy.Array
            Array containing the observed event times in ascending order.
        """
        self.his = history

    def exp_intensity(self, sum_less_equal=True):
        """
        Calculate the (exponential) intensity of a HawkesN process
        symbolically.

        Parameters
        ----------
        sum_less_equal : bool, default: True
            If True, we sum over all event times <= time t. Otherwise, we sum
            over all event times < time t.

        Returns
        -------
        exp_intensity_ : sympy.core.mul.Mul
            A sympy expression containing the symbols scale, decay, n, and t.
        """
        scale, decay, n, t = symbols("scale decay n t")
        events_until_t = sum(
            [Piecewise((1, h <= t), (0, True)) for h in self.his]
        )
        return (1 - events_until_t/n) * scale * decay * sum(
            [Piecewise(
                (exp(-decay * (t - h)), h <= t if sum_less_equal else h < t),
                (0, True)
             ) for h in self.his])

    def plot_exp_intensity(self, t_max, scale, decay, n, step=0.01,
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
        scale : float
            Scale parameter of the exponential intensity.
        decay : float
            Decay parameter of the exponential intensity.
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
            This arg is used in :meth:`self.exp_intensity`.
        """
        subs_list = [("scale", scale), ("decay", decay), ("n", n)]
        exp_intensity = self.exp_intensity(
            sum_less_equal=sum_less_equal).subs(subs_list)
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
        if self.his is not None and scale is not None and decay is not None \
                and n is not None:
            title += " with event history \{" \
                     + ",".join(str(i) for i in self.his[:4]) \
                     + (", ..." if len(self.his) > 4 else "") \
                     + "\} \nand parameters: scale=" + str(scale) \
                     + ", decay=" + str(decay) + ", $N$=" + str(n)
        title += "."
        plt.title(title)

        if fname is not None:
            plt.savefig(fname + ".pdf")

    def llf(self, sum_less_equal=True):
        """
        Parameters
        ----------
        sum_less_equal : bool, default: True
            This arg is used in :meth:`self.exp_intensity`.

        Returns
        -------
        llf : sympy.core.add.Add
            The log-likelihood function as symbolic expression (containing the
            symbols `scale`, `decay`, and `n`).
        """
        scale, decay, n = symbols("scale decay n")
        intensity = self.exp_intensity(sum_less_equal)

        # for h in self.his:
        #     print("intensity at", h, "is:", intensity.subs("t", h))
        first_event = len(self.his) - sum(1 for t in self.his if t > 0)
        his_pos = self.his[first_event:]
        addend_sum = sum(log(intensity.subs("t", h)) for h in his_pos)
        # print("SUM PART", addend_sum.subs([("scale", .5), ("decay", .5), ("n", 100)]))

        i, j = symbols("i j", integer=True)
        addend_int = scale * sum(
            (n - (i + 1)) / n * (
                exp(-decay * (self.his[i] - self.his[j]))
                -
                exp(-decay * (self.his[i + 1] - self.his[j]))
            )
            for i in range(len(self.his)-1)
            for j in range(i+1))
        # print("INT PART", addend_int.subs([("scale", .5), ("decay", .5), ("n", 100)]))
        return addend_sum - addend_int

    def llf_gradient(self, sum_less_equal=True):
        """
        Calculate the gradient of the log-likelihood function symbolically.

        Parameters
        ----------
        sum_less_equal : bool, default: True
            This arg is passed to :meth:`self.llf`.

        Returns
        -------
        gradient : sympy.Array
            An array containing three entries. The first (second) [third] entry
            is the derivative of the log-likelihood function w.r.t. the scale
            (decay) [N] parameter.
        """
        scale, decay, n = symbols("scale decay n")
        return derive_by_array(self.llf(sum_less_equal), [scale, decay, n])

    def fit(self, scale_start, decay_start, n_start):
        """
        Parameters
        ----------
        scale_start : float
            Starting value for the likelihood maximization.
        decay_start : float
            Starting value for the likelihood maximization.
        n_start : float
            Starting value for the likelihood maximization.

        Returns
        -------
        TODO
        """
        llf_sym = self.llf()
        llf_grad_sym = self.llf_gradient()
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
            result = llf_sym.subs([("scale", scale_decay_n[0]),
                                 ("decay", scale_decay_n[1]),
                                 ("n", scale_decay_n[2])])
            print("llf", result)
            return result

        def negative_llf_gradient(scale_decay_n):
            result = -llf_grad_sym.subs([("scale", scale_decay_n[0]),
                                         ("decay", scale_decay_n[1]),
                                         ("n", scale_decay_n[2])])
            print("-grad:", result)
            return np.array(result, dtype=np.float64)

        eps = np.finfo(float).eps

        return fmin_l_bfgs_b(
            func=negative_llf,  # minimize this
            x0=np.array([scale_start, decay_start, n_start]),  # initial guess
            fprime=negative_llf_gradient,
            bounds=[(eps, None), (eps, None), (len(self.his), None)],
            iprint=101
        )

    @staticmethod
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

    @classmethod
    def size_distribution(cls, scale=5, decay=0.2, n=400, history=None,
                              transition_matrix=None):
        sir_params = cls.to_sir(scale=scale, decay=decay, n=n)

        if history is None:
            return StochasticSIR.size_distribution(
                **sir_params, transition_matrix=transition_matrix)
        else:
            max_time = history[-2]
        # TODO


