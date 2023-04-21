#####
# Finding the piece-wise hermite polynomial interpolation of a set of points
# MPI version
#####
from time import time
import numpy as np
from mpi4py import MPI

from mpi4py import MPI

# comm = MPI.COMM_WORLD

# my_rank = comm.Get_rank()
# p = comm.Get_size()

# if my_rank != 0:
#     message = 'Hello from the other rank {}'.format(my_rank)
#     comm.send(message, dest = 0)

# else:
#     for pid in range(1,p):
#         message = comm.recv(source = pid)
#         print('Process {} receives message: {}'.format(my_rank, message))

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

buffer = np.array([0])

if rank == 0:
    #when r is 0, send the value to r is 1
    data = buffer
    comm.Send(data, dest = 1)
    
    #receives the value from N-1
    comm.Recv(buffer, source = size - 1)
    print('Processor name is', name, "rank:", rank, 'and', "value:", buffer[0])


# if rank >= 1:
#     #the value received by (r-1)
#     comm.Recv(buffer, source = rank-1)
    
#     #it squares its rank  r adds the result  r^2 to the value of its own buffer
#     buffer = buffer+rank**2
    
#     #the value of its own buffer, and then sends the sum to Process r+1
#     if rank != (size - 1):
#         comm.Send(buffer, dest = rank+1)
#     if rank == (size - 1):
#         comm.Send(buffer, dest = 0)

for i in range(size-1):
    comm.Recv(buffer, source = rank-1)
    buffer = buffer+rank**2
    comm.Send(buffer, dest = (rank+1)% size)
    
    
    
