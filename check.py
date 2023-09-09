import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from data import X, Y
from math import *

def check(coeff: list, x_list, y_list):
    (a, b, c, d, e, f) = coeff
    M = np.array([[a, b/2], [b/2, c]])
    evals, evecs = linalg.eigh(M)

    P = evecs
    P_inv = linalg.inv(P)
    D = np.diag(evals)

    d_, e_ = np.round(np.matmul(np.array([d, e]), P), 1)

    for i in range(len(x_list)):
        pt = np.array([x_list[i], y_list[i]])
        pt = np.matmul(P_inv, pt)
        x_list[i], y_list[i] = pt
    
    plot(x_list, y_list, 2)

    coeff = np.array([evals[0], 0, evals[1], d_, e_, f])
    (a, b, c, d, e) = [evals[0], evals[1], d_, e_, f]
    # print(evals[0], evals[1], d_, e_, f)  new coeff

    x_shift = c / (2*a)
    f -= c**2 / (4*a)
    y_shift = d / (2*b)
    f -= d**2 / (4*b)

    # print(a, x_shift, b, y_shift, f)

    for i in range(len(x_list)):
        x_list[i] -= x_shift
        y_list[i] -= y_shift

    plot(x_list, y_list, 3)
    coeff = np.array([a, b, f])
    # print(a, b, f)

    # Now we have parametrization (acos(t), bcos(t))
    a = sqrt(-f / a)
    b = sqrt(-f / b)

def plot(X, Y, i):
    plt.title(str(i))
    plt.scatter(X, Y)
    plt.axvline(0)
    plt.axhline(0)
    # plt.scatter(pts1[:, 0], pts1[:, 1])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

plot(X, Y, 1)
check([1, 3, 5, 7, 9, 11], X, Y)