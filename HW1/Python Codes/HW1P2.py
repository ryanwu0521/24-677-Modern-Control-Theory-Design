# 24-677 Linear Control System Homework 1 Exercise 2 Part 2 [Ryan Wu]
import numpy as np
import matplotlib.pyplot as plt

P1 = P2 = P3 = 0.1  # initial conditions
alpha = 1.2  # [-]
gamma = 3  # [-]
sigma = 0.1  # [-]

G = np.asarray([[1.0, 0.2, 0.1],
                [0.1, 2.0, 0.1],
                [0.3, 0.1, 3.0]])


def simulate(P1, P2, P3, alpha, gamma, sigma, G, iter):
    A = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            if i != j:
                A[i][j] = alpha * gamma * G[i][j] / G[i][i]

    B = np.zeros([3, 1])
    for i in range(3):
        B[i][0] = alpha * gamma * (sigma ** 2) / G[i][0]

    P = np.array([[P1], [P2], [P3]])
    S = [
        [G[0][0] * P[0] / (sigma ** 2 + G[0][1] * P[1] + G[0][2] * P[2])],
        [G[1][1] * P[1] / (sigma ** 2 + G[1][0] * P[0] + G[1][2] * P[2])],
        [G[2][2] * P[2] / (sigma ** 2 + G[2][0] * P[0] + G[2][1] * P[1])]
    ]

    P_array = [P]
    S_array = [S]

    for _ in range(iter):
        P = np.dot(A, P) + B
        P_array.append(P)

        S = [
            [G[0][0] * P[0] / (sigma ** 2 + G[0][1] * P[1] + G[0][2] * P[2])],
            [G[1][1] * P[1] / (sigma ** 2 + G[1][0] * P[0] + G[1][2] * P[2])],
            [G[2][2] * P[2] / (sigma ** 2 + G[2][0] * P[0] + G[2][1] * P[1])]
        ]
        S_array.append(S)

    return P_array, S_array


def plot_result(result, type, x_name, y_name, title):
    n = len(result)

    line1, line2, line3 = list(), list(), list()
    for i in range(n):
        line1.append(result[i][0][0])
        line2.append(result[i][1][0])
        line3.append(result[i][2][0])

    time = list(range(n))

    plt.plot(time, line1, label=f"{type}1")
    plt.plot(time, line2, label=f"{type}2")
    plt.plot(time, line3, label=f"{type}3")

    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.legend()
    plt.show()


p_array, s_array = simulate(P1=0.1, P2=0.1, P3=0.1, alpha=1.2, gamma=3, sigma=0.1, G=G, iter=20)
plot_result(p_array, "P", "time", "Pi", "P1=P2=P3=0.1, gamma=3")
plot_result(s_array, "S", "time", "Si", "P1=P2=P3=0.1, gamma=3")

p_array, s_array = simulate(P1=0.1, P2=0.1, P3=0.1, alpha=1.2, gamma=5, sigma=0.1, G=G, iter=20)
plot_result(p_array, "P", "time", "Pi", "P1=P2=P3=0.1, gamma=5")
plot_result(s_array, "S", "time", "Si", "P1=P2=P3=0.1, gamma=5")

p_array, s_array = simulate(P1=0.1, P2=0.01, P3=0.02, alpha=1.2, gamma=3, sigma=0.1, G=G, iter=20)
plot_result(p_array, "P", "time", "Pi", "P1=0.1, P2=0.01, P3=0.02, gamma=3")
plot_result(s_array, "S", "time", "Si", "P1=0.1, P2=0.01, P3=0.02, gamma=3")

p_array, s_array = simulate(P1=0.1, P2=0.01, P3=0.02, alpha=1.2, gamma=5, sigma=0.1, G=G, iter=20)
plot_result(p_array, "P", "time", "Pi", "P1=0.1, P2=0.01, P3=0.02, gamma=5")
plot_result(s_array, "S", "time", "Si", "P1=0.1, P2=0.01, P3=0.02, gamma=5")