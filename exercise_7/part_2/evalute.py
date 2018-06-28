import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')

from part_1.jacoby import Jacoby


def build_matrix(dim=None, periodic=False):
    if not dim:
        raise ValueError("Please enter a Dimension for the matrix")
    dim_sq = dim ** 2
    last_row = dim_sq - dim
    matrix = 4 * np.identity(dim_sq, dtype=float)  # initialize j==k with -4

    if periodic:
        j = 0
        while j < dim_sq:
            if j == 0:
                # upper left corner
                matrix[j][dim - 1] = -1  # -1 Term
                matrix[j][1] = -1  # +1 Term
                matrix[j][dim] = -1  # +N Term
                matrix[j][last_row] = -1  # -N Term

            elif j == dim - 1:
                # lower left corner
                matrix[j][j - 1] = -1  # -1 Term
                matrix[j][0] = -1  # +1 Term
                matrix[j][j + dim] = -1  # +N Term
                matrix[j][dim_sq - 1] = -1  # -N Term

            elif j == last_row:
                # uper right corner
                matrix[j][dim_sq - 1] = -1  # -1 Term
                matrix[j][j + 1] = -1  # +1 Term
                matrix[j][0] = -1  # +N Term
                matrix[j][j - dim] = -1  # -N Term

            elif j == dim_sq - 1:
                # lower right corner
                matrix[j][j - 1] = -1  # -1 Term
                matrix[j][last_row] = -1  # +1 Term
                matrix[j][dim - 1] = -1  # +N Term
                matrix[j][j - dim] = -1  # -N Term

            elif 0 < j < dim - 1:
                # left border
                matrix[j][last_row + j] = -1  # the left partner (j-N)
                matrix[j][j + dim] = -1
                matrix[j][j + 1] = -1
                matrix[j][j - 1] = -1
                pass
            elif dim_sq > j > last_row:
                # rigth border
                matrix[j][j - dim] = -1
                matrix[j][j - last_row] = -1  # the right partner (j+N)
                matrix[j][j + 1] = -1
                matrix[j][j - 1] = -1
            elif not j % dim:
                # top border
                matrix[j][j - dim] = -1
                matrix[j][j + dim] = -1
                matrix[j][j + 1] = -1
                matrix[j][j + dim - 1] = -1  # the top partner (j-1)

            elif not (j + 1) % dim:
                # bottom border
                matrix[j][j - dim] = -1
                matrix[j][j + dim] = -1
                matrix[j][j - dim + 1] = -1  # the bottom partner
                matrix[j][j - 1] = -1
            else:
                matrix[j][j - 1] = -1
                matrix[j][j + 1] = -1
                matrix[j][j + dim] = -1
                matrix[j][j - dim] = -1
            j += 1

    return matrix


class BuildMatrix:

    def __init__(self, dim, periodic=False):

        self.dim = dim
        self.dim_sq = dim ** 2
        self.last_row = self.dim_sq - self.dim
        self.periodic = periodic
        self.index_func = self.get_index_func()
        self.matrix = self.build_matrix()

    def get_index_func(self):
        if self.periodic:
            return self.real_periodic_index
        return self.real_non_periodic_index

    def build_matrix(self):
        matrix = np.zeros((self.dim_sq, self.dim_sq))
        j = 0

        while j < self.dim_sq:
            # produces the correct indexes
            indexes = self.get_indexes(j)
            for index in indexes:
                if index == "x":
                    continue
                matrix[j][index] = -1
            matrix[j][j] = np.count_nonzero(matrix[j] == -1)
            j += 1
        return matrix

    def get_indexes(self, j):
        d = self.index_func(j, -1)
        f = self.index_func(j, 1)
        g = self.index_func(j, -self.dim)
        h = self.index_func(j, self.dim)
        return d, f, g, h

    def real_periodic_index(self, j, add):
        if -1 <= add <= 1:
            # the problem with -1 and 1 are the upper and lower boundaries. So we only need to look at those cases
            if not j % self.dim and add == -1:
                # upper boundary only -1 is a problem
                return j + self.dim - 1

            if not (j + 1) % self.dim and add == 1:
                # lower boundary
                return j - self.dim + 1
            return j + add
        else:
            # it is obviously dim . the problem with +dim and - dim are the right and left border.
            if j <= self.dim - 1 and add < 0:
                return self.last_row + j
            if j >= self.last_row and add > 0:
                return j - self.last_row
            return j + add

    def real_non_periodic_index(self, j, add):
        # we return "x" if the index shall not be set.
        if -1 <= add <= 1:
            # the problem with -1 and 1 are the upper and lower boundaries. So we only need to look at those cases
            if not j % self.dim and add == -1:
                # upper boundary only -1 is a problem
                return "x"

            if not (j + 1) % self.dim and add == 1:
                # lower boundary
                return "x"
            return j + add
        else:
            # it is obviously dim . the problem with +dim and - dim are the right and left border.
            if j <= self.dim - 1 and add < 0:
                return "x"
            if j >= self.last_row and add > 0:
                return "x"
            return j + add


def evaluate_x(index, eigen_vectors, eigen_values, time):
    i = 0
    y = np.zeros(49)
    ret_arr = []
    for t in time:
        while i < 49:
            eigen = eigen_vectors[:, i]
            y += eigen * eigen[0]*np.cos(np.sqrt(eigen_values[i][i])*t)
            i += 1
        ret_arr.append(y[index])
        y = np.zeros(49)
        i = 0
    return ret_arr


# matrix for periodic Boundaries
# matrix_class = BuildMatrix(dim=4, periodic=True).matrix

# matrix for None periodic boundaries
# matrix_class = BuildMatrix(dim=4, periodic=False).matrix

# matrix for None periodic boundaries with 7 dim


matrix_class = BuildMatrix(dim=7, periodic=False).matrix
time = np.arange(0.0, 30.0, step=0.1)


solved = Jacoby(matrix=matrix_class, error=1e-20)

solved.calculate()

print(solved.eigen_vectors[:, 1])

for j in [0, 6, 24, 48]:
    x = evaluate_x(index=j, eigen_vectors=solved.eigen_vectors, eigen_values=solved.matrix, time=time)
    plt.plot(time, x, ".", label="j={}".format(j))

plt.legend(loc="best")
plt.show()
# print(solved.eigen_vectors)
#
# x = []
#
# for j in range(16):
#     x.append((solved.matrix[j][j]))
#
# x.sort()
#
# print(x)




# x = [
#     np.array([0.32, +0.32, +0.32, +0.32, +0.32, +0.32, +0.32, +0.32, +0.32, +0.32]),
#     np.array([-0.07, +0.20, -0.32, +0.40, -0.44, +0.44, -0.40, +0.32, -0.20, +0.07]),
#     np.array([-0.44, -0.40, -0.32, -0.20, -0.07, +0.07, +0.20, +0.32, +0.40, +0.44]),
#     np.array([+0.26, -0.43, -0.00, +0.43, -0.26, -0.26, +0.43, +0.00, -0.43, +0.26]),
#     np.array([+0.36, -0.14, -0.45, -0.14, +0.36, +0.36, -0.14, -0.45, -0.14, +0.36]),
#     np.array([+0.14, -0.36, +0.45, -0.36, +0.14, +0.14, -0.36, +0.45, -0.36, +0.14]),
#     np.array([+0.43, +0.26, +0.00, -0.26, -0.43, -0.43, -0.26, -0.00, +0.26, +0.43]),
#     np.array([-0.20, +0.44, -0.32, -0.07, +0.40, -0.40, +0.07, +0.32, -0.44, +0.20]),
#     np.array([+0.32, -0.32, -0.32, +0.32, +0.32, -0.32, -0.32, +0.32, +0.32, -0.32]),
#     np.array([-0.40, -0.07, +0.32, +0.44, +0.20, -0.20, -0.44, -0.32, +0.07, +0.40]),
# ]
