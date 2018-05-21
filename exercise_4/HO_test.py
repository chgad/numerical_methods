from shooting_method import ShootingMethod
import numpy as np


def ho_function(y, x):
    # This resambles (x² -E)* psi(x)
    print(" This is x:", x)
    return (x**2 - y[2]) * y[0]


ho_test = ShootingMethod(tau=0.01, force=ho_function, r=1.0, rdot=0.0, range=500, energy=0.8, file="HO_test.csv")

ho_test.shooting_procedure()

print("the energy is :",ho_test.energy)
