import pyopencl as cl
import numpy as np
from time import time
from python_functions.utils import memory_usage_psutil
import psutil, os


# Random data
k = 70
s = 100
total_bases = (k * s)
h_SK_4 = np.random.choice(range(0,4), total_bases).astype("uint8").reshape((s,k))

s = h_SK_4.shape[0] # Number of kmers
k = h_SK_4.shape[1] # Kmer size

# Compute output columns
cSK_10_32  =  (k-1)/16 + 1

# Output matrix
h_SK_10_32 = np.ndarray((s, cSK_10_32)).astype(np.uint32)

print "#############  KMERS BASE 4 A DECIMAL (32bits) K = {}, S = {}  #################".format(k, s)

# OpenCL definitions
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/N_B32.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
N_B32 = programa.N_B32
N_B32.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32])

# Copy input data from host to device
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)

# Output buffer
d_SK_10_32 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_10_32.nbytes)

# Execution Range
rango_global = (k, s)
rango_local = (k,1)

# Kernel Execution
print "Executing kernel"
t1 = time()
N_32(cola, rango_global, rango_local, d_SK_4, d_SK_10_32, k, s, cSK_10_32 )
cola.finish()
print "Kernel took {} seconds in the execution".format(time()-t1)

# Retrieve output
cl.enqueue_copy(cola, h_SK_10_32, d_SK_10_32)

cola.finish()
