import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import sys

sys.path.append('..')

from chi_sq import ChiSquareApprox

file = sys.argv[1]

V = []
delta_V = []
r = []


with open(file, 'r') as f:
    reader = csv.reader(f, delimiter=";")
    for row in reader:
        r.append(float(row[0]))
        V.append(float(row[1]))
        delta_V.append(float(row[2]))
    f.close()

V = np.array(V[2:])
delta_V = np.array(delta_V[2:])
r = np.array(r[2:])


def g_1(x):
    return 1


def g_2(x):
    return 1 / x


def g_3(x):
    return x


fitting = ChiSquareApprox(x_data=r, y_data=V, y_errors=delta_V, g_of_x_components=[g_1, g_2, g_3])

fac1,fac2, fac3 = fitting.determine_components()[0]

x_set = np.arange(1.0,max(r), step=0.01)

plt.errorbar(r, V, delta_V, ecolor="r", fmt=".b", label="Computed data")
plt.plot(x_set, fac1*g_1(x_set) + fac2*g_2(x_set) + fac3*g_3(x_set))
plt.legend(loc="best")

plt.show()
