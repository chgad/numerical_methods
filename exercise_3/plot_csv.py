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

circle_x = np.cos(np.arange(0.0, 2*np.pi, 0.01))
circle_y = np.sin(np.arange(0.0, 2*np.pi, 0.01))
print(max(y))

plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
axes = plt.gca()
axes.set_xlim([-2.0,2.0])
axes.set_ylim([-2.0,2.0])
plt.plot(x, y, label="RK 4th Order")
plt.plot(circle_x, circle_y, label="Circle")
# plt.plot(tau, np.array(r)-cosine(np.array(tau)), label="Absolute difference computed phi and cosine")
plt.legend(loc="best")
plt.show()
