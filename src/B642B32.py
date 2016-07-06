import pyopencl as cl
import numpy as np
from time import time
import scipy

# BASE 10 A BASE 4
# Input Matrix
SK_2_64 = [[4294967296,2],[1,8589934591]]
h_SK_2_64 =np.ndarray((len(SK_2_64), len(SK_2_64[0]))).astype(np.uint64)
h_SK_2_64[:] = SK_2_64

#Output matrix
h_SK_2_32 =np.ndarray((len(SK_2_64), len(SK_2_64[0])*2)).astype(np.uint32)

print h_SK_2_32
cSK_2_64 = h_SK_2_64.shape[1]
cSK_2_32 = h_SK_2_32.shape[1]
s = h_SK_2_64.shape[0] # Total kmers

print "#############  KMERS BASE 2 (64BITS) A BASE 2 (32 BITS)  #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B642B32.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B642B32 = programa.B642B32
B642B32.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32])

# Copy input data from host to device
d_SK_2_64 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_64)

# Output buffer
d_SK_2_32 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_32.nbytes)

# Execution Range
rango_global = (h_SK_2_32.shape[1], s)

# Kernel Execution
B642B32(cola, rango_global, None, d_SK_2_64, d_SK_2_32, cSK_2_64, cSK_2_32, s)

cola.finish()
# Retrieve output
cl.enqueue_copy(cola, h_SK_2_32, d_SK_2_32)

print h_SK_2_64
print h_SK_2_32
