import numpy as np
import sys
sys.path.append('../')

from exercise_2.runge_kutta import RungeKutta4th
from exercise_3.adaptive_runge_kutta import AdaptiveRungeKutta4th


def kepler_force(y):
    return -1 * np.power(np.square(y[0]).sum(axis=0), -3/2) * y[0]

# PART 1

# for t in [10.0,1.0,0.9,0.8,0.3,0.01]:
#     kepler = RungeKutta4th(tau=t, force=kepler_force, r=np.array([1.0, 0.0]), rdot=np.array([0.0, 1.0]),
#                            range=10000, file="kepler_no_{}.csv".format(t))
#
#     kepler.solve_OED()

# Runge Kutta 4th ORder method with adaptive steps
# Question remains : How do i now for how long the program has to run ?

# PART2

# v)
# kepler = AdaptiveRungeKutta4th(tau=0.01, force=kepler_force, r=np.array([1.0, 0.0]), rdot=np.array([0.0, 1.0]),
#                                range=7000, file="kepler_error_{0}.csv".format(0.01), adaptive=True, abs_error=0.01)
# kepler.solve_OED()

# vii)


excen1, excen2 = (0.1, 0.9)

a1, a2 = (1.1, 1.9)

b1, b2 = np.sqrt(1-excen1**2), np.sqrt(1-excen2**2)

p1, p2 = (1-excen1**2), (1-excen2**2)

v1, v2 = np.sqrt(p1)/a1, np.sqrt(p2)/a2


# Compute with excentricity 0.1
kepler = AdaptiveRungeKutta4th(tau=0.01, force=kepler_force, r=np.array([-a1, 0]), rdot=np.array([0.0, -v1]),
                               range=7500, file="kepler_excentr_{0}.csv".format(0.1), adaptive=True, abs_error=0.01)
kepler.solve_OED()


# Compute with excentricity 0.9
kepler = AdaptiveRungeKutta4th(tau=0.01, force=kepler_force, r=np.array([-a2, 0.0]), rdot=np.array([0.0, -v2]),
                               range=15000, file="kepler_excentr_{0}.csv".format(0.9), adaptive=True, abs_error=0.01)

kepler.solve_OED()
