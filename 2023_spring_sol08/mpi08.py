#####
# Finding the piece-wise hermite polynomial interpolation of a set of points
# MPI version
#####
from time import time
import numpy as np
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

buffer = np.array([0])

if rank >= 1:
    #the value received by (r-1)
    comm.Recv(buffer, source = rank - 1)
    
    #it squares its rank  r adds the result  r^2 to the value of its own buffer
    buffer = buffer+rank**2
    
    #the value of its own buffer, and then sends the sum to Process r+1
    if rank != (size - 1):
        comm.Send(buffer, dest = rank + 1)
    else:
        comm.Send(buffer, dest = 0)
else:
    #for r = 0, first send its value to r = 1
    comm.Send(buffer, dest = 1)
    
    #receive the value from process N-1
    comm.Recv(buffer, source = size - 1)
    print("rank:", rank, 'and', "value:", buffer[0])
    
