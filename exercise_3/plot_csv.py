import csv
import sys

import numpy as np
import math
import matplotlib.pyplot as plt

file = sys.argv[1]
row1, row2 = int(sys.argv[2]), int(sys.argv[3])


x = []
y = []


with open(file, 'r') as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        x.append(float(row[row1]))
        y.append(float(row[row2]))
    f.close()


# Prove the second Kepler Law
# if L IS constant the Integral of L dt is the same for any 2 time diferences dt_1 = dt_2
# a realy lazymethodproduce the radial Vektor, which for elipsis changes periodically with time
# and integrate it over the discreat time steps tau

def produce_r(l):
    return np.power(np.square(l).sum(axis=1), 1 / 2)


r = produce_r(list(zip(x,y)))

L1 = np.trapz(r[:1001], dx=0.01)
L2 = np.trapz(r[5000:6001], dx=0.01)

print("L1", L1)
print("L2", L2)

circle_x = np.cos(np.arange(0.0, 2*np.pi, 0.01))
circle_y = np.sin(np.arange(0.0, 2*np.pi, 0.01))
print(max(y))

plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
axes = plt.gca()
axes.set_xlim([-2.0,2.0])
axes.set_ylim([-2.0,2.0])
plt.plot(x, y, label="RK 4th Order")
# plt.plot(circle_x, circle_y, label="Circle")
# plt.plot(tau, np.array(r)-cosine(np.array(tau)), label="Absolute difference computed phi and cosine")
plt.legend(loc="best")
plt.show()

