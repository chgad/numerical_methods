import numpy as np
import sys
sys.path.append('../')

from part_1.jacoby import Jacoby

def build_matrix(dim=None, periodic=False):
    if not dim:
        raise ValueError("Please enter a Dimension for the matrix")
    dim_sq = dim **2
    last_row = dim_sq -dim
    matrix = 4*np.identity(dim_sq, dtype=float) # initialize j==k with -4

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
        self.dim_sq = dim**2
        self.last_row = self.dim_sq - self.dim
        self.periodic = periodic
        self.matrix = self.build_matrix()

    def construct_diagonal(self):
        return 4*np.identity(self.dim_sq, dtype=float)

    def build_matrix(self):
        matrix = self.construct_diagonal()
        j=0

        while j < self.dim_sq:
            if self.periodic:
                # produces the correct indexes
                indexes= self.get_indexes(j)
                for index in indexes:
                    matrix[j][index] = -1
                j += 1
        return matrix

    def get_indexes(self,j):
        d = self.real_periodic_index(j, -1)
        f = self.real_periodic_index(j, 1)
        g = self.real_periodic_index(j,-self.dim)
        h = self.real_periodic_index(j,self.dim)
        return d, f, g, h

    def real_periodic_index(self, j,add):
        if -1 <= add <= 1:
            # the problem with -1 and 1 are the upper and lower boundaries. So we only need to look at those cases
            if not j%self.dim and add == -1 :
                # upper boundary only -1 is a problem
                return j + self.dim - 1

            if not (j+1)%self.dim and add == 1:
                # lower boundary
                return j - self.dim + 1
            return j + add
        else:
            # it is obviously dim . the problem with +dim and - dim are the right and left border.
            if j <= self.dim - 1 and add < 0:
                return self.last_row + j
            if j >= self.last_row and add >0:
                return j - self.last_row
            return j + add


matrix = build_matrix(dim=4, periodic=True)
matrix_class = BuildMatrix(dim=4, periodic=True).matrix

if np.array_equal(matrix, matrix_class):
    print("WoHOOOOOOO")
else:
    for i in matrix_class:
        print(i)
        input()


