import numpy as np


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


class Jacoby:

    def __init__(self, matrix, error):

        self.matrix = matrix
        self.dim = self.matrix.shape[0]
        self.p = 0  # first row off diagonal
        self.q = 1  # first column off diagonal
        self.error = error
        self.eigen_vectors = np.identity(self.dim)

    def change_off_diagonals(self):

        if not(self.q < self.dim-1):
            # if we are at the end of a row
            self.p += 1
            self.q = self.p + 1

            if self.q == self.dim:
                # reinitialize
                self.p = 0
                self.q = 1
        else:
            self.q += 1

    def calculate_theta(self):
        return (self.matrix[self.q][self.q] - self.matrix[self.p][self.p])/(2.0*self.matrix[self.p][self.q])

    def calculate_tan(self, theta):
        return sign(theta)/(np.abs(theta) + np.power(theta**2+1.0, 0.5))

    def calculate_rotation_stuff(self):
        theta = self.calculate_theta()

        return self.calculate_tan(theta=theta)

    def calculate_tau_sin(self, tan):
        tan_x_half = tan / (1.0 + np.power(1.0 + tan ** 2, 0.5))
        sin = 2.0*tan_x_half/(1+tan_x_half**2)
        cos = (1-tan_x_half**2)/(1+tan_x_half**2)
        return tan_x_half, sin, cos

    def compute_diagonals(self, p, q, tan):
        self.matrix[p][p] -= tan * self.matrix[p][q]
        self.matrix[q][q] += tan * self.matrix[p][q]

    def set_p_q_element_zero(self, p, q):
        self.matrix[p][q] = 0.0
        self.matrix[q][p] = 0.0

    def compute_rest_elements(self, p, q, sin, tau):
        k = 0
        while k < self.dim:
            if k == q or k == p:
                k += 1
                continue
            old_a_k_p = self.matrix[k][p]

            self.matrix[p][k] -= sin * (self.matrix[k][q] + tau * self.matrix[k][p])
            self.matrix[k][p] = self.matrix[p][k]

            self.matrix[q][k] += sin * (old_a_k_p - tau * self.matrix[k][q])
            self.matrix[k][q] = self.matrix[q][k]
            k += 1

    def compute_deviation(self):
        k = 0
        l = 0
        S = 0.0
        while k <= self.dim - 1:
            while l <= self.dim - 1:
                if l == k:
                    l += 1
                    continue
                S += self.matrix[k][l]**2
                l += 1
            k += 1

        return S

    def compute_eigen_vectors(self, p, q, sin, tau):
        k = 0
        while k < self.dim:
            old_eigen_k_p = self.eigen_vectors[k][p]
            self.eigen_vectors[k][p] -= sin * (self.eigen_vectors[k][q] + tau * self.eigen_vectors[k][p])
            self.eigen_vectors[k][q] += sin * (old_eigen_k_p - tau * self.eigen_vectors[k][q])
            k += 1

    def calculate(self):
        tan = 0
        s = 1000.0
        while s > self.error:
            p = self.p
            q = self.q

            if self.matrix[p][q] == 0.0:
                self.change_off_diagonals()
                continue

            tan = self.calculate_rotation_stuff()
            tau, sin, cos = self.calculate_tau_sin(tan)
            # calculate new diagonals and set the p_q q_p elements to Zero
            self.compute_diagonals(p=p, q=q, tan=tan)
            self.set_p_q_element_zero(p=p, q=q)
            # compute A_k_p
            self.compute_rest_elements(p=p, q=q, sin=sin, tau=tau)
            self.compute_eigen_vectors(p=p, q=q, sin=sin, tau=tau)
            s = self.compute_deviation()
            self.change_off_diagonals()


# Test stuff

# test_matrix = np.array([[2.0, 1.0],[1.0, 2.0]])
#
# thre_b_thre = np.array([[3.0,2.0,1.0],[2.0,1.0,0.0],[1.0,0.0,3.0]])

# test = np.array(
#     [[4.0, 1.0, 3.0, 1.0],
#      [1.0, 6.0, 12.0, 11.0],
#      [3.0, 12.0, 3.0, 5.0],
#      [1.0, 11.0, 5.0, 9.0]])
#
# solving = Jacoby(thre_b_thre, error=1e-20)
#
# solving.calculate()
# print("Eigenvectors: \n",solving.eigen_vectors)
# print("Eigenvalues: \n", solving.matrix)
# thre_b_thre = np.array([[3.0,2.0,1.0],[2.0,1.0,0.0],[1.0,0.0,3.0]])
# print("Eigenvalue = {:.6}".format(solving.matrix[1][1]), "And corresponding eigen vektor", solving.eigen_vectors[:, 1]/solving.eigen_vectors[:,1][2])
# print(solving.eigen_vectors[:, 1] * solving.matrix[1][1], "=", np.matmul(thre_b_thre, solving.eigen_vectors[:, 1]))
