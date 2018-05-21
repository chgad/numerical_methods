import csv
import sys

import numpy as np
import math
import matplotlib.pyplot as plt

file = sys.argv[1]
row1, row2 = int(sys.argv[2]), int(sys.argv[3])


x = []
y = []
tau = []

with open(file, 'r') as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        x.append(float(row[row1]))
        y.append(float(row[row2]))
    f.close()

# x = np.arange(-100, 100,step=0.1)
#
#
# def potential(x,l):
#     return 0.5 * x**2 + l * x**4
#
#
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$")
# axes = plt.gca()
# axes.axhline(color='r')
# axes.set_xlim([-5, 5])
# axes.set_ylim([0.0, 1000])
# for l in [1000, 10000, 1e6, 1e7, 1e8]:
#     plt.plot(x, potential(x,l), label=r"Potential wiht $\lambda$={}".format(l))


plt.plot(x, y, label="RK4th ORder")
plt.legend(loc="best")
plt.show()

