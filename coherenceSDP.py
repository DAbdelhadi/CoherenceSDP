import cvxpy as cp
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def contraction_SDP(n,p):
    d = 2**n #n: number of qubits

    eta = cp.Variable(1)
    W = cp.Variable((2*d,2*d))

    # Diagonal matrix which is free to be optimized over
    Delta = cp.diag(cp.Variable((1,d)))

    #Dephasing channel matrix with diagonal removed
    M = np.matrix([[(1-p)**bin(i^j).count('1') for i in range(d)]for j in range(d)])-np.eye(d)
    objective = cp.Minimize(eta)

    constraints = [cp.diag(W)-eta*np.ones(2*d)<=0,
                   W>>0,
                   W[0:d,d:2*d]==M+Delta,
                   W[d:2*d,0:d]==cp.transpose(M+Delta)
                   ]

    prob = cp.Problem(objective, constraints)
    # The optimal objective value is returned by `prob.solve()`.
    prob.solve()
    np.set_printoptions(precision=2)

    return eta.value
