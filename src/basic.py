import numpy as np


class Problem:
    """
    A class representing a problem instance for graph similarity or matching.
    """

    def __init__(self, mat_A: np.ndarray, mat_B: np.ndarray, vec_A: np.ndarray, vec_B: np.ndarray, subgraph: bool=True):
        """
        Initialize the Problem instance with adjacency matrices and vectors.
        subgraph (bool): Full graph match / subgraph match.
        """
        self.mat_A = mat_A
        self.mat_B = mat_B
        self.vec_A = np.array(vec_A)
        self.vec_B = np.array(vec_B)
        self.subgraph = True
        self.N = len(mat_A[0, :])
        self.M = len(mat_B[0, :])
        self.L = len(bin(self.M - 1)) - 2
        assert mat_A.shape == (self.N,self.N)
        assert mat_B.shape == (self.M,self.M)
        assert self.vec_A.shape == (self.N,)
        assert self.vec_B.shape == (self.M,)


    def result_to_f(self, result: str):
        """
        Convert 01 string result from quantum circuit to f
        """
        result = result[::-1]
        assert len(result) == self.N*self.L
        blocks = [result[i * self.L : (i + 1) * self.L] for i in range(self.N)]
        f = [int(block, 2) for block in blocks]
        return f
    

    def valid(self, f: list):
        """
        Decide whether f is a injection from [N] to [M]
        """
        assert len(f) == self.N
        for i in range(self.N):
            assert f[i] >= 0 and f[i] < 2**self.L
            if (f[i]) >= self.M:
                return False
            for j in range(i + 1, self.N):
                if f[i] == f[j]:
                    return False
        return True
    

    def eval_W(self, f: list, l1, l2):
        """
        Evaluate W(f)
        """
        W = 0
        assert len(f) == self.N
        for i in range(self.N):
            assert f[i] >= 0 and f[i] < 2**self.L
            if f[i] >= self.M:
                W += l1
            for j in range(i, self.N):
                if j > i and f[i] == f[j]:
                    W += l2
                if f[i] < self.M and f[j] < self.M:
                    W += (self.mat_A[i][j] - self.mat_B[f[i]][f[j]]) ** 2
        return W

    def eval_d(self, f: list):
        """
        Evaluate d(f), assert f is valid
        """
        d2 = 0
        assert self.valid(f)
        for i in range(self.N):
            for j in range(i, self.N):
                d2 += (self.mat_A[i][j] - self.mat_B[f[i]][f[j]]) ** 2
        d = np.sqrt(d2)
        return d
    
    def brutal_force(self):
        """
        Classical brutal-force solver, retures d_min, solutions
        """
        d_min = 1000
        solutions = []
        f = [0]*self.N
        for x in range(self.M**self.N):
            for i in range(self.N):
                f[i] = (x // self.M**i) % self.M
            if self.valid(f):
                d_value = self.eval_d(f)
                if d_value < d_min:
                    d_min = d_value
                    solutions = [f]
                elif d_value == d_min:
                    solutions += f
        return d_min, solutions