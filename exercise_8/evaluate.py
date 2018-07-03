import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

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

print("V", V)
print("r", r)
print("dV", delta_V)




plt.errorbar(r, V, delta_V, ecolor="r",fmt=".b", label="Computed data")
plt.legend(loc="best")

plt.show()