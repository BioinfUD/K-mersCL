import pyopencl as cl
import numpy as np
from time import time
from python_functions.utils import memory_usage_psutil
import psutil, os


# Random data
k = 48
s = 1000000
total_bases = (k * s)
h_SK_4 = np.random.randint(4, size=total_bases).astype("uint8").reshape((s,k))

s = h_SK_4.shape[0] # Number of kmers
k = h_SK_4.shape[1] # Kmer size

# Compute output columns
cSK_10_64  =  (k-1)/32 + 1

# Output matrix
h_SK_10_64 = np.ndarray((s, cSK_10_64)).astype(np.uint64)

print "#############  KMERS BASE 4 A DECIMAL (64bits) K = {}, S = {}  #################".format(k, s)

# OpenCL definitions
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/N_64.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
N_B64 = programa.N_B64
N_B64.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32])

# Copy input data from host to device
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)

# Output buffer
d_SK_10_64 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_10_64.nbytes)

# Execution Range
rango_global = (k, s)
rango_local = (k,1)


# Kernel Execution
print "Executing kernel"
t1 = time()
N_B64(cola, rango_global, rango_local, d_SK_4, d_SK_10_64, k, s, cSK_10_64 )
cola.finish()
print "Kernel took {} seconds in the execution".format(time()-t1)

# Retrieve output
cl.enqueue_copy(cola, h_SK_10_64, d_SK_10_64)

cola.finish()
