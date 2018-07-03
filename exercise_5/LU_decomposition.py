import numpy as np


class LUDecomposition:

    def __init__(self, matrix, vectors, pivot=None):
        self.matrix = matrix
        self.vectors = vectors
        self.validate()
        self.dim = self.matrix.shape[0]
        self.U = np.zeros((self.dim,self.dim))
        self.L = np.identity(self.dim)
        self.y = np.zeros(self.vectors.shape)
        self.x = np.zeros(self.vectors.shape)
        self.pivot = pivot

    def validate(self):
        """
        We want to besure that the Matrix is Quadratic and that all vectors have the same dimension
        as the Matrix.
        :return:
        """
        matrix_shape = self.matrix.shape
        vectors_shape = self.vectors.shape

        assert matrix_shape[0] == matrix_shape[1]
        assert vectors_shape[1] == matrix_shape[1]

    def pivot(self):
        if not self.pivot:
            pass
        else:
            pass

    def matrix_multiplikation(self, limit=0, column=0, row=0, usecase=None, vector= None, l=0):
        """

        :param limit:
        :param column:
        :param row:
        :param usecase: 0 --> L * U
                        1 --> L * y
                        2 --> U * x
        :return:
        """
        uses = {0: [self.L, self.U],
                1: [self.L, vector],
                2: [self.U, vector]}
        mult1, mult2 = uses[usecase]

        if limit < 0:
            return 0
        summ = 0.0
        if not usecase:
            while l <= limit:
                summ += mult1[row][l] * mult2[l][column]
                l += 1
        else:
            while l <= limit:
                summ += mult1[row][l] * mult2[l]
                l += 1
        return summ

    def vector_matrix_multiplication(self, limit=0, row=0, y=None):
        if limit < 0:
            return 0
        summ = 0
        k = 0
        while k <= limit:
            summ += self.L[row][k]*y[k]
            k += 1

        return summ

    def decomposition(self):
        """
        This method decomposes the Matrix A into two Matrises L and U such that
        A = L U holds.
        :return:
        """
        k = 0
        while k <= self.dim - 1:
            j = 0
            while j <= k:
                self.U[j][k] = self.matrix[j][k] - self.matrix_multiplikation(row=j, column=k, limit=j-1, usecase=0)
                j += 1

            j = k+1
            while j <= self.dim - 1:
                self.L[j][k] = (self.matrix[j][k] -
                                self.matrix_multiplikation(row=j, column=k, limit=j-1, usecase=0))/self.U[k][k]
                j += 1

            k += 1

    def solve_y(self):
        """
        This method solves Ax = LUx = Ly = b where we only compute  Ux = y
        :return:
        """
        j = 0

        while j <= self.dim - 1:
            for y, b in zip(self.y, self.vectors):
                y[j] = b[j] - self.matrix_multiplikation(limit=j-1, row=j, vector=y, usecase=1)
            j += 1

    def solve_x(self):
        j = self.dim - 1
        while j >= 0:
            for x, y in zip(self.x, self.y):
                x[j] = (y[j] - self.matrix_multiplikation(limit=self.dim-1, l=j+1, usecase=2, row=j, vector=x))\
                       / self.U[j][j]
            j -= 1


# matrix = np.array([[0.6, 0.13, 1.28], [0.0123, 0.078, 0.97], [0.5, 0.47, 0.718]])
#
# matrix = np.random.rand(10, 10)
# vector = np.full((1, 10), 1.0)
matrix =np.array([[3,6],[6,14]])
vector = np.array([[ 905.55555556,1466.66666667]])
# solving = LUDecomposition(matrix=matrix, vectors=vector)
# solving.decomposition()
# solving.solve_y()
# solving.solve_x()
# print(solving.matrix)
# print("upper \n", solving.U)
# print("lower \n", solving.L)
# print("y \n", solving.y)
# print("x \n", solving.x)

# print(np.matmul(matrix, solving.x[0])-vector[0])
