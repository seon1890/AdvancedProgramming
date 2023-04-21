
from mpi4py import MPI
import numpy as np
import statistics


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

n = 256000

if rank == 0:
    data = np.random.random(n).astype('f').reshape(size, int(n/size))    
#     data = float(np.random.random(n).reshape(size, int(n/size)))  
    
if rank != 0:
    data = None

zeros = np.zeros(int(n/size), dtype='f')


comm.Scatter(data,zeros, root = 0)

data = comm.bcast(data, root=0)
comm.Bcast(data, root=0)

# n = size * 2
# m = 5
# if rank == 0:
#     A = np.random.randn(n, m)
#     x = np.random.randn(m)

#     ytarget = A.dot(x) # to check the result
    
#     A = A.reshape(size, -1, m)
# else:
#     A = None
#     x = np.zeros(m)

# Asmall = np.zeros((n // size, m))

# # Asmall = comm.scatter(A, root=0)
# comm.Scatter(A, Asmall, root=0)

# # x = comm.bcast(x, root=0)
# comm.Bcast(x, root=0)

# print('rank', rank, ':', Asmall.shape, x.shape)

globalsums = comm.allreduce(sum(zeros), MPI.SUM)
avg = globalsums / n


diff = sum(map(lambda i: (i - avg)**2, zeros))

globaldiff = comm.reduce(diff, MPI.SUM, 0)

print(rank, sum(zeros))
# print('==================================================================')
if rank == 0:
    std = np.sqrt(globaldiff / n)
#     print('==================================================================')
    print('Mean = ', avg, 'Standard deviation', std)
else:
    None
