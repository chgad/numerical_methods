import numpy as np
import sys
sys.path.append('../')

from exercise_2.runge_kutta import RungeKutta4th


class AdaptiveRungeKutta4th(RungeKutta4th):

    def __init__(self, adaptive=False, abs_error=0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive = adaptive
        self.abs_error = abs_error
        self.tau_range = [0.0]
        self.recenttime = self.tau
        self.allowed_tau = [0.2, 5.0]
        self.delta_range = []

    def perform_two_steps(self):
        # print("this is tau halfs:", self.tau)
        value = None
        for i in range(2):
            value = self.perform_one_step()
        return value

    def compute_delta(self, y_tau, y_tau_halfs):
        return sum(np.linalg.norm(y_tau_halfs-y_tau, ord=np.inf, axis=0))/15

    def compute_new_tau(self, calculated_delta, tau):
        prefactor = 0.9 * (self.abs_error/calculated_delta)**(1/5)

        # clamp the new tau to [0.2*tau, 5.0*tau] to avoid toooooo small steps
        if prefactor <= self.allowed_tau[0]:
            prefactor = 0.2
        elif prefactor >= self.allowed_tau[1]:
            prefactor = 5.0
        return prefactor * tau

    def solve_OED(self):
        if self.adaptive:
            # if adaptive is set do the adaptive stepsize procedure
            i = 1
            while i <= self.range:
                one_tau_y = self.perform_one_step()
                t = float(self.tau)
                self.tau /= 2.0
                # perform two steps with tau halfes
                two_tau_halfs_y = self.perform_two_steps()
#                 print("step {0}; this is y_tau and y_tau_halfs",one_tau_y, two_tau_halfs_y)
                # calculate delta and new tau
                self.tau = t
                delta = self.compute_delta(y_tau=one_tau_y, y_tau_halfs=two_tau_halfs_y)
                new_tau = self.compute_new_tau(calculated_delta=delta, tau=t)
                # if tau half was a good option we need to reduce our time evolution parameter tau
                if delta <= self.abs_error:
                    self.y = two_tau_halfs_y
                    self.y_range.append(self.y[0])
                    self.tau_range.append(self.recenttime)
                    self.recenttime += t
                    self.tau = new_tau

                # else redo the same with the new tau
                else:
                    self.tau = new_tau
                    self.recenttime = self.tau
                    i -= 1

                i += 1

            self.write_results_to_csv(file_name=self.file_name)
        else:
            self.tau_range = np.arange(0.0, self.range * self.tau, self.tau)
            super().solve_OED()


def pendulum_force(y):
    """
    Y is a 2D array of 2 * N Dim : Y = [[x1, x2, x3 ..., xN] ,[xdot1, xdot2, xdot3, ..., xdotN]]
    :param y:
    :return: 1* N Dim array
    """
    return -1*y[0]


# None adaptive
first_try = AdaptiveRungeKutta4th(tau=0.001, abs_error=1e-12, force=pendulum_force, r=np.array([0.0000001, ]), rdot=np.array([0.0, ]),
                                  range=100000, file="first_try.csv", adaptive=False)
first_try.solve_OED()

print(first_try.combine_rows()[1])


