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
from math import sin


class RungeKutta4th:
    def __init__(self, tau=0.1, force=None, r=None , rdot=None, range=None, file=""):

        self.tau = tau
        self.force = force
        self.r = r
        self.rdot = rdot
        self.y = np.array([r, rdot])
        self.f = self.compute_f()

        self.range = range
        self.file_name = file
        self.validate()

        # if everything is correct initalize the Runge-Kutta k's

        self.k1 = np.array([])
        self.k2 = np.array([])
        self.k3 = np.array([])
        self.k4 = np.array([])

        # Tau and computed y values
        self.tau_range = np.arange(0.0, self.range, self.tau)
        self.y_range = [self.r, ]

    def validate(self):
        if not self.force:
            raise ValueError('Please provide a Force for computatuion')
        if not self.r:
            raise ValueError('Please provide Starting conditions for r')
        if self.rdot is None:
            raise ValueError('Please provide Starting conditions for rdot')
        if not (self.tau and self.tau > 0.0):
            raise ValueError('Please provide a tau > 0')

    def compute_f(self):
        return np.array([self.y[1], self.force(self.y[0])])

    def compute_k1(self):
        self.k1 = self.compute_f() * self.tau

    def compute_k2(self):
        help_f = np.array([self.y[1] + 0.5*self.k1[1], self.force(self.y[0] + 0.5*self.k1[0])])

        self.k2 = help_f * self.tau

    def compute_k3(self):
        help_f = np.array([self.y[1] + 0.5 * self.k2[1], self.force(self.y[0] + 0.5 * self.k2[0])])
        self.k3 = help_f * self.tau

    def compute_k4(self):
        help_f = np.array([self.y[1] + self.k3[1], self.force(self.y[0] + self.k3[0])])
        self.k4 = help_f * self.tau

    def compute_all_k(self):
        self.compute_k1()
        self.compute_k2()
        self.compute_k3()
        self.compute_k4()

    def compute_y(self):
        self.y = self.y + 1.0/6.0 * (self.k1 + 2*self.k2 + 2*self.k3 + self.k4)

    def write_results_to_csv(self,file_name):
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(self.tau_range, self.y_range))
            f.close()

    def solve_OED(self):
        i = 1
        while i <= self.range:
            # compute all k's for preparation to compute the new y
            self.compute_all_k()
            # compute new y
            self.compute_y()
            # append y to the list of y values
            self.y_range.append(self.y[0])
            i += 1
        self.write_results_to_csv(file_name=self.file_name)


def pendulum_force(x):
    return -1*sin(x)


first_try = RungeKutta4th(force=pendulum_force, r=0.00001, rdot=0.0, range=1000, file="first_try.csv")

first_try.solve_OED()
