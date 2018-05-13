import numpy as np
import sys
sys.path.append('../')

from exercise_2.runge_kutta import RungeKutta4th
from exercise_3.adaptive_runge_kutta import AdaptiveRungeKutta4th

def kepler_force(y):
    return -1 * np.power(np.square(y[0]).sum(axis=0), -3/2) * y[0]


# for t in [10.0,1.0,0.9,0.8,0.3,0.01]:
#     kepler = RungeKutta4th(tau=t, force=kepler_force, r=np.array([1.0, 0.0]), rdot=np.array([0.0, 1.0]),
#                            range=10000, file="kepler_no_{}.csv".format(t))
#
#     kepler.solve_OED()

kepler = AdaptiveRungeKutta4th(tau=0.001, force=kepler_force, r=np.array([1.0, 0.0]), rdot=np.array([0.0, 1.0]),
                               range=100000, file="kepler_no_{}.csv".format(0.01), adaptive=True, abs_error=0.01)
kepler.solve_OED()