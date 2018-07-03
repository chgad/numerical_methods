import csv
import sys

import numpy as np
import math
import matplotlib.pyplot as plt

file = sys.argv[1]
row1, row2 = int(sys.argv[2]), int(sys.argv [3])
prefactor = float(sys.argv[4])
print( row1,row2,prefactor)
tau = []
r = []


def cosine(x):
    return np.cos(math.sqrt(1)*x)*prefactor


with open(file, 'r') as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        tau.append(float(row[row1]))
        r.append(float(row[row2]))
    f.close()

print(max(tau))
plt.xlabel(r"$\tau$")
plt.ylabel(r"$\phi$")

plt.plot(tau, r, label="RK 4th Order")
plt.plot(tau, cosine(np.array(tau)), label="Analytical solution")
# plt.plot(tau, np.array(r)-cosine(np.array(tau)), label="Absolute difference computed phi and cosine")
plt.legend(loc="best")
plt.show()
