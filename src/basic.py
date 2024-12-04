import numpy

def get_NML(A, B):
  N = len(A[0,:])
  M = len(B[0,:])
  L = len(bin(M-1)) - 2
  return N, M, L

# Evaluate W(f)
def eval_W(f, A, B, L1, L2):
  w = 0
  return w

# Evaluate d(f)
def eval_d(f, A, B):
  d = 0
  return d

# Evaluate SIM(f)
def eval_SIM(f, A, B):
  return 1 / (1 + eval_d(f, A, B))