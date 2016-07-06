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
h_SK_2_16 =np.ndarray((len(SK_2_64), len(SK_2_64[0])*4)).astype(np.uint16)

print h_SK_2_16
cSK_2_64 = h_SK_2_64.shape[1]
cSK_2_16 = h_SK_2_16.shape[1]
s = h_SK_2_64.shape[0] # Total kmers

print "#############  KMERS BASE 2 (64BITS) A BASE 2 (16 BITS)  #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B642B16.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B642B16 = programa.B642B16
B642B16.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32])

# Copy input data from host to device
d_SK_2_64 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_64)

# Output buffer
d_SK_2_16 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_16.nbytes)

# Execution Range
rango_global = (h_SK_2_16.shape[1], s)

# Kernel Execution
B642B16(cola, rango_global, None, d_SK_2_64, d_SK_2_16, cSK_2_64, cSK_2_16, s)

cola.finish()
# Retrieve output
cl.enqueue_copy(cola, h_SK_2_16, d_SK_2_16)


print h_SK_2_64
print h_SK_2_16
