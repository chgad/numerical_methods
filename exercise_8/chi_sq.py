import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')

from exercise_5.LU_decomposition import LUDecomposition

class ChiSquareApprox:

    def __init__(self,x_data=None,y_data=None, y_errors=None, g_of_x_components=None):
        self.x_data = x_data
        self.y_data = y_data
        self.y_errors = y_errors**2
        self.g_comps = g_of_x_components
        self.A = self.build_matrix()
        self.weights = self.build_weights()

    def build_matrix(self):

        matrix = np.zeros((len(self.x_data), len(self.g_comps)))

        for fk_ind, fkt in list(enumerate(self.g_comps)):
            for x_ind, x in list(enumerate(self.x_data)):
                matrix[x_ind][fk_ind] = fkt(x)

        return matrix

    def build_weights(self):
        length = len(self.y_errors)
        matrix = np.identity(length)
        j = 0
        while j < length:
            matrix[j][j] = 1 / self.y_errors[j]
            j += 1

        return matrix


    def determine_components(self):
        transpose_mat = np.matrix.transpose(self.A)

        vectors_mat = np.matmul(transpose_mat, self.weights)
        vector = np.matmul(vectors_mat, self.y_data)

        matrix = np.matmul(np.matmul(transpose_mat, self.weights), self.A)
        solved = LUDecomposition(matrix=matrix, vectors=np.array([vector]))
        solved.decomposition()
        solved.solve_y()
        solved.solve_x()
        return solved.x

#
# def g_1(x):
#     return 1
#
#
# def g_2(x):
#     return x
#
#
# x = np.array([0, 1, 2, 3])
#
# y = np.array([2, 5, 8, 11])
# y_error = np.array([1.1, 1.45, 0.503, 0.137])
#
# test = ChiSquareApprox(x_data=x, y_data=y, y_errors=y_error, g_of_x_components=[g_1, g_2])
#
# fac1, fac2 = test.determine_components()[0]
# print(fac1,fac2)
# plt.errorbar(x, y, yerr=y_error, fmt=".r", label="data")
#
# x_synt = np.arange(0.0,3.0,0.01)
#
# plt.plot(x_synt,fac1*g_1(x_synt)+fac2*g_2(x_synt), label="fit")
#
# plt.show()
