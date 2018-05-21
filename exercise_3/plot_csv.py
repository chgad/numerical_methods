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
        # x.append(float(row[row1]))
        y.append(float(row[row2]))
        tau.append(float(row[0]))
    f.close()

print(tau[-1])
# Case of 0 excentricity --> check for working algorithm
# circle_x = np.cos(np.arange(0.0, 2*np.pi, 0.01))
# circle_y = np.sin(np.arange(0.0, 2*np.pi, 0.01))


# PART 1
# Prove the second Kepler Law
# if L IS constant the Integral of L dt is the same for any 2 time diferences dt_1 = dt_2
# a realy lazymethodproduce the radial Vektor, which for elipsis changes periodically with time
# and integrate it over the discreat time steps tau
#
# def produce_r(l):
#     return np.power(np.square(l).sum(axis=1), 1 / 2)
#

# r = produce_r(list(zip(x,y)))
#
# L1 = np.trapz(r[:1001], dx=0.01)
# L2 = np.trapz(r[5000:6001], dx=0.01)
#
# print("L1", L1)
# print("L2", L2)


# PART 2
# Checkaverage Tau
# steps = 7500
# if "0.9" in sys.argv[0]:
#     steps = 15000
# print("The Average Tau was : {}".format(tau[-1]/steps))

# Chechk third Kepler Law
# find index where y coordinate becomes positive for first time
#  this equals a half period
# get the corresponding time and multiply by two

# index = 1
# while y[index] < -sys.float_info.min:
#     index += 1
#
# half_period = tau[index]
#
# period = half_period * 2
#
# print(period)
#
# excen1, excen2 = (0.1, 0.9)
#
# a1, a2 = (1.1, 1.9)
#
# b1, b2 = a1*np.sqrt(1-excen1**2), a2*np.sqrt(1-excen2**2)
#
# p1, p2 = b1, b2
#
# v1, v2 = np.sqrt(p1)/a1, np.sqrt(p2)/a2
#
#
#
# a1, t1 = 1, 14.30154545586655
# a2,t2 = 1, 13.531666219993632
# print("(a1/a2)³ = {0}, (T1/T2)² = {1}".format((a1/a2)**3, (t1/t2)**2))


plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
axes = plt.gca()
axes.axhline(color='r')
# axes.set_xlim([-2.0,2.0])
# axes.set_ylim([-2.0,2.0])
plt.plot(tau, y, label="RK 4th Order")
# plt.plot(tau, y, label="y(t)")
# plt.plot(circle_x, circle_y, label="Circle")
plt.plot(tau, 0.0000001*np.cos(np.array(tau)), label="Analytical")
plt.legend(loc="best")
plt.show()

