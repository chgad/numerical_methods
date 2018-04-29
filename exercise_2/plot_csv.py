import csv

import numpy as np
import math
import matplotlib.pyplot as plt

file = "first_try.csv"
tau = []
r = []


def cosine(x):
    return np.cos(math.sqrt(1)*x)*0.00001


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
# plt.plot(tau, np.array(r)-cosine(np.array(tau)), label="Absolute difference computed phi and cosine")
plt.show()
