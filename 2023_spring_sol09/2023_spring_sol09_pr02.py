
import numpy as np
from numpy import pi, exp, sqrt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# def integral(a_r, h, n):
#     integ = 0.0
#     for j in range(n):
#         t = a_r + (j + 0.5) * h
#         integ += sin(t) * h
#     return integ

def integral(a_r, h, n, K):
    
    integ = np.zeros(K)
    for j in range(n):
        t = a_r + (j + 0.5) * h
        integs = []
        for i in range(1, K+1):
            integs.append(t**i*exp(-t))
        integ = integ+np.array(integs)
    integ = h*integ
    return integ

K = 16
M = 1000

# a = 0.0
# b = numpy.pi / 2
# dest = 0
# my_int = numpy.zeros(1)
# integral_sum = numpy.zeros(1)

integral_sum = np.zeros(16)
n = np.array(1000)


h = M / (n * size) 
ar = rank * h * n

my_int = integral(ar, h, n, K)

comm.Reduce(my_int, integral_sum, MPI.SUM, root=0)

# print("Process ", rank, " has the partial integral ", my_int[0])

# Initialize value of n only if this is rank 0
if rank == 0:
    for i in range(K):
        print((i, integral_sum[i]))
