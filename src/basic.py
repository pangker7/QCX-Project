import numpy as np


def get_NML(A, B):
    N = len(A[0, :])
    M = len(B[0, :])
    L = len(bin(M - 1)) - 2
    return N, M, L


# Convert 01 string result from quantum circuit to f
def result_to_f(result, A, B):
    N, _, L = get_NML(A, B)
    result = result[::-1]
    assert len(result) == N * L
    blocks = [result[i * L : (i + 1) * L] for i in range(N)]
    f = [int(block, 2) for block in blocks]
    return f


# Determine whether f is a valid injection from [N] to [M]
def valid(f, A, B):
    N, M, L = get_NML(A, B)
    assert len(f) == N
    for i in range(N):
        assert f[i] >= 0 and f[i] < 2**L
        if (f[i]) >= M:
            return False
        for j in range(i + 1, N):
            if f[i] == f[j]:
                return False
    return True


# Evaluate W(f)
def eval_W(f, A, B, L1, L2):
    W = 0
    N, M, L = get_NML(A, B)
    assert len(f) == N
    for i in range(N):
        assert f[i] >= 0 and f[i] < 2**L
        if f[i] >= M:
            W += L1
        for j in range(i, N):
            if j > i and f[i] == f[j]:
                W += L2
            if f[i] < M and f[j] < M:
                W += (A[i][j] - B[f[i]][f[j]]) ** 2
    return W


# Evaluate d(f), assert f is valid
def eval_d(f, A, B):
    d2 = 0
    N, _, _ = get_NML(A, B)
    assert valid(f, A, B)
    for i in range(N):
        for j in range(i, N):
            d2 += (A[i][j] - B[f[i]][f[j]]) ** 2
    d = np.sqrt(d2)
    return d


# Evaluate SIM(f), assert f is valid
def eval_SIM(f, A, B):
    return 1 / (1 + eval_d(f, A, B))
