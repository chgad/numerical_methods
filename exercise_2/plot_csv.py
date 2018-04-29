import csv
import sys

import numpy as np
import math
import matplotlib.pyplot as plt

file = sys.argv[1]
pi = float(sys.argv[2])/4.0 * np.pi
tau = []
r = []


def cosine(x):
    return np.cos(math.sqrt(1)*x)*pi


with open(file, 'r') as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        tau.append(float(row[0]))
        r.append(float(row[1]))
    f.close()

plt.xlabel(r"$\tau$")
plt.ylabel(r"$\phi$")

plt.plot(tau, r, label="RK 4th Order")
plt.plot(tau, cosine(np.array(tau)), label="Analytical solution")
plt.title(r"$\phi_0 = ${}$\pi$".format(float(sys.argv[2])/4.0))
# plt.plot(tau, np.array(r)-cosine(np.array(tau)), label="Absolute difference computed phi and cosine")
plt.legend(loc="best")
plt.show()
