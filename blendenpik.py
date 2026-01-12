"""Python implementations of sketch-and-precondition solvers for linear systems."""

import numpy as np
from scipy.fftpack import dct
from scipy.linalg import qr
from scipy.sparse.linalg import lsqr


def blendenpik(A, b, sketch_size) -> np.ndarray:
    """Blendenpik solver for overdetermined linear systems Ax = b.

    See https://pdos.csail.mit.edu/~petar/papers/blendenpik-v1.pdf

    Parameters
    ----------
    A : ndarray
        The input matrix of shape (m, n) with m >> n.
    b : ndarray
        The right-hand side vector of shape (m,).
    sketch_size : int
        The number of rows to sample.

    Returns
    -------
    x : ndarray
        The solution vector of shape (n,).

    """
    m, n = A.shape

    # get diagonal matrix with 50/50 +1/-1 values
    D = np.diag(np.random.choice([-1, 1], m))

    M = dct(D @ A)

    # select random rows of M
    S = np.random.randint(0, m, sketch_size)
    SM = M[S, :]

    _, R = qr(SM, overwrite_a=True, mode='economic')
    R_inv = np.linalg.inv(R)

    y = lsqr(A @ R_inv, b)[0]
    x = lsqr(R, y)[0]

    return x


if __name__ == '__main__':
    A = np.random.sample((20000, 100))
    b = np.random.sample(20000)

    direct = lsqr(A, b)[0]
    bpk = blendenpik(A, b, sketch_size=400)

    #print(direct)
    #print(bpk)
    #print(saa)

    print(np.linalg.norm(A @ direct - b) / np.linalg.norm(A @ bpk - b))
