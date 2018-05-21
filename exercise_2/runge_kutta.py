# This File contains an implementation of the Runge-Kutta algorithm
# for solving Ordinary Differentail Equations

# The Given Problem to compute the solution for is the "simple" pendulum
# with a mass m , length l which is only affected by the gravitational Force (g Gravitaionla accelaration)


# The Equations of Motion for the angle p is given by:
# d²/dt² p = g/l sin(p)

# For testing we apply the small angle approximation p << 1
# for which sin(p) ~= p and the ODE can be solved analytically

# p(t=0) = p_0
# d/dt p(t=0) = 0

# Approximation:
# The solution is then given by p(t) = p_0 * cos(sqrt(g/l)*t)

import numpy as np
import csv


class RungeKutta4th:
    def __init__(self, tau=0.1, force=None, r=None, rdot=None, range=None, file="", output=True):
        # if we want to create an output file output must be true.
        self.output = output

        self.tau = tau
        # the force needs to take a 2 D array and handle it correctly
        self.force = force
        # r and rdot are expected to be numpy arrays
        self.r = r
        self.rdot = rdot
        # y will become a pseudo matrix 2 D array
        self.y = np.array([r, rdot])

        self.range = range
        self.file_name = file
        # if everything is correct initalize the Runge-Kutta k's

        self.k1 = np.array([])
        self.k2 = np.array([])
        self.k3 = np.array([])
        self.k4 = np.array([])

        # Tau and computed y values
        self.tau_range = np.arange(0.0, self.range*self.tau, self.tau)
        self.y_range = [self.r, ]

    def compute_f(self, y):
        return np.array([y[1], self.force(y)])

    def compute_k1(self):

        self.k1 = self.compute_f(self.y) * self.tau

    def compute_k2(self):

        self.k2 = self.compute_f(self.y + 0.5*self.k1) * self.tau

    def compute_k3(self):

        self.k3 = self.compute_f(self.y + 0.5*self.k2) * self.tau

    def compute_k4(self):
        self.k4 = self.compute_f(self.y + self.k3) * self.tau

    def compute_all_k(self):
        self.compute_k1()
        self.compute_k2()
        self.compute_k3()
        self.compute_k4()

    def compute_y(self):
        return self.y + 1.0/6.0 * (self.k1 + 2*self.k2 + 2*self.k3 + self.k4)

    def combine_rows(self):
        values = []
        h = 0
        for t in self.tau_range:
            values.append(np.insert(self.y_range[h], 0, t))
            h += 1
        return values

    def write_results_to_csv(self,file_name):
        rows = self.combine_rows()
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            f.close()

    def perform_one_step(self):
        # compute all k's for preparation to compute the new y
        self.compute_all_k()
        # compute new y
        return self.compute_y()

    def solve_OED(self):
        i = 1
        while i <= self.range:
            self.y = self.perform_one_step()
            # append y to the list of y values
            self.y_range.append(self.y[0])
            i += 1
        if self.output:
            self.write_results_to_csv(file_name=self.file_name)


def pendulum_force(y):
    """
    Y is a 2D array of 2 * N Dim : Y = [[x1, x2, x3 ..., xN] ,[xdot1, xdot2, xdot3, ..., xdotN]]
    :param y:
    :return: 1* N Dim array
    """
    return -1*y[0]


# first_try = RungeKutta4th(force=pendulum_force, r=np.array([0.00001, 0.00002, ]), rdot=np.array([0.0,0.0 ]),
#                           range=1000, file="first_try.csv")
# first_try.solve_OED()

# k = 1
# for i in [np.pi /4.0, np.pi/2.0, 3.0/4.0 * np.pi]:
#     runge= RungeKutta4th(force=pendulum_force, r=i, rdot=0.0, range=400, file="runge_kutta_{}pi.csv".format(k/4.0))
#     runge.solve_OED()
#     k += 1