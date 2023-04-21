
import sys
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

p = 1
x, y = float(sys.argv[1]), float(sys.argv[2])
buffer = np.array([x, y, p])


for i in range(5):
    if rank == 0:
        # multiply p by x
        buffer[2] *= buffer[0]
        
        # send the message to process 1 through channel 0
        comm.Send(buffer, dest = 1, tag = 0)
        
        # receive the message from process 1 through channel 1
        comm.Recv(buffer, source = 1, tag = 1)
        
        if i == 4:
            print(f"final value of p: {buffer[2]}")
    else:
        # receive the message from process 0 through channel 0
        comm.Recv(buffer, source = 0, tag = 0)
        
        # divide p by y
        buffer[2] /= buffer[1]
        
        # send the message to process 0 through channel 1
        comm.Send(buffer, dest = 0, tag = 1)
