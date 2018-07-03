import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')

from exercise_5.LU_decomposition import LUDecomposition

class ChiSquareApprox:

    def __init__(self,x_data=None,y_data=None, y_errors=None, g_of_x_components=None):
        self.x_data = x_data
        self.y_data = y_data
        self.y_errors = y_errors
        self.g_comps = g_of_x_components
        self.A = self.build_matrix()

    def build_matrix(self):

        matrix = np.zeros((len(self.x_data), len(self.g_comps)))

        for fk_ind, fkt in list(enumerate(self.g_comps)):
            for x_ind, x in list(enumerate(self.x_data)):
                matrix[fk_ind][x_ind] = fkt(x)

        return matrix

    def determine_components(self):
        pass
