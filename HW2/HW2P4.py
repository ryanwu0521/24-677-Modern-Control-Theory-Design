import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import trapz

# Given variables
x_0 = 0
t = 1
t_0 = 0
k = 5
m = 0

# Setting up matrices
A = np.array([[0, 1],
              [-2, -2]])

B = np.array([[1],
              [1]])

C = np.array([[2, 3]])

D = 0

# Compute eigenvalues
eigenvalues = np.linalg.eig(A)
eigenvalues = eigenvalues[0]
print("Eigenvalues:", eigenvalues)

# Solving e^At
exp_At = expm(A * t)
print("eAt", exp_At)

# Uncertain about the tau in the integral
tau = 1
def u(tau):

    return np.array(np.sin(tau)).reshape(1, 1)


t_intervals = np.linspace(t_0, t, 100)


# CT Calculation (Wasn't able to solve)
# y_t = C @ np.exp(A * (t - t_0)) * x_0 + C @ np.array([trapz(np.exp(A * (t - tau)) @ B @ u(tau), t_intervals) for tau in t_intervals]) + D

# DT Calculation
def get_y(k):
    return C @ np.linalg.matrix_power(A, k - m - 1) @ B

print("DT system for y(5):", get_y(k))
