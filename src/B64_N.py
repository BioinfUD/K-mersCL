import pyopencl as cl
import numpy as np
from time import time
from python_functions.utils import memory_usage_psutil
import psutil, os


# Parameters
k = 48
s = 1000000

# Compute output columns
cSK_10_64  =  (k-1)/32 + 1

# Random data
total_decimals = (cSK_10_64 * s)
h_SK_10_64 = np.random.randint(2**63, size=total_decimals).astype("uint64").reshape((s,cSK_10_64))

# Fill first column with valid values
if (k%32!=0):
    h_SK_10_64[:,0] =  np.random.randint(2**(k%32), size=s)

# Output matrix
h_SK_4 = np.ndarray((s, k)).astype(np.uint8)

print "#############  KMERS DECIMAL (64bits) a BASE 4  K = {}, S = {}  #################".format(k, s)

# OpenCL definitions
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B64_N.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B64_N = programa.B64_N
B64_N.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32])

# Copy input data from host to device
d_SK_10_64 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_10_64)

# Output buffer
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_4.nbytes)

# Execution Range
rango_global = (k, s)
rango_local = (k,1)


# Kernel Execution
print "Executing kernel"
t1 = time()
B64_N(cola, rango_global, rango_local, d_SK_4, d_SK_10_64, k, s, cSK_10_64 )
cola.finish()
print "Kernel took {} seconds in the execution".format(time()-t1)

# Retrieve output
cl.enqueue_copy(cola, h_SK_4, d_SK_4)

cola.finish()
