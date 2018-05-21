#  This Skript is for build an algorithm for solving boundary-value Problems
# numerically. It utilizes the so call shooting method, in combination with
# the Runge-Kutta method for solving differential equations.


import numpy as np
import sys
import csv
sys.path.append('../')

from exercise_2.runge_kutta import RungeKutta4th


class ShootingMethod:

    def __init__(self,energy=0.0, max_accuracy=1e-6, output=False,
                 tau=0.1, force=None, r=None, rdot=None, range=None, file=""):

        self.output = output

        self.tau = tau
        self.recenttime = tau
        # the force needs to take a 2 D array and handle it correctly
        self.force = force
        # r and rdot are expected to be numpy arrays
        self.r = r
        self.rdot = rdot
        # y will become a pseudo matrix 2 D array
        self.y = np.array([r, rdot, energy])

        self.range = range
        self.file_name = file
        # if everything is correct initalize the Runge-Kutta k's

        self.k1 = np.array([])
        self.k2 = np.array([])
        self.k3 = np.array([])
        self.k4 = np.array([])

        # Tau and computed y values
        self.tau_range = np.arange(0.0, self.range * self.tau, self.tau)
        self.y_range = [self.r, ]

        self.energy = energy
        self.max_accuracy = max_accuracy
        self.derivative_step = 1e-7

        # time scale
        print("The computation will begin at the value 0.0 and end at {} for the evolution parameter"
              .format(self.tau*self.range))

    def compute_f(self, y):
        return np.array([y[1], self.force(y, x=self.recenttime), 0.0])

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
            self.recenttime += self.tau
            i += 1
        if self.output:
            self.write_results_to_csv(file_name=self.file_name)

    def combine_rows(self):
        values = []
        h = 0
        for t in self.tau_range:
            values.append([t, self.y_range[h]])
            h += 1
        return values

    def write_results_to_csv(self,file_name):
        rows = self.combine_rows()
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            f.close()

    def restore_inital(self):
        self.y = np.array([self.r, self.rdot, self.energy])
        self.y_range = [self.r, ]
        self.recenttime = self.tau

    def raphson_improvement(self, plus_h, minus_h, psi):
        derivative = (plus_h - minus_h)/(2.0*self.derivative_step)
        return psi / derivative

    def one_improvement_shot(self):

        # evalutate with energy parameter reduced by a derivative step
        self.y[-1] = self.energy - self.derivative_step
        self.solve_OED()
        psi_e_minus_h = self.y[0]
        self.restore_inital()
        # evalutate with energy parameter increased by a derivative step
        self.y[-1] = self.energy + self.derivative_step
        self.solve_OED()
        psi_e_plus_h = self.y[0]
        self.restore_inital()
        # evaluate Trajectory with inital energy
        self.solve_OED()
        psi_e = self.y[0]

        return psi_e_minus_h, psi_e, psi_e_plus_h

    def shooting_procedure(self):
        while True:
            psi_e_minus_h, psi_e, psi_e_plus_h = self.one_improvement_shot()
            delta = self.raphson_improvement(plus_h=psi_e_plus_h, minus_h=psi_e_minus_h, psi=psi_e)
            if abs(delta) < self.max_accuracy:
                break
            self.energy -= delta
            self.restore_inital()

        self.write_results_to_csv(file_name=self.file_name)


