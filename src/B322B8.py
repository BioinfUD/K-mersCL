import pyopencl as cl
import numpy as np
from time import time
import scipy

# BASE 10 A BASE 4
# Input Matrix
SK_2_32 = [[4294967295,2],[1,131071]]
h_SK_2_32 =np.ndarray((len(SK_2_32), len(SK_2_32[0]))).astype(np.uint32)
h_SK_2_32[:] = SK_2_32


#Output matrix
h_SK_2_8 =np.ndarray((len(SK_2_32), len(SK_2_32[0])*4)).astype(np.uint8)
cSK_2_32 = h_SK_2_32.shape[1]
cSK_2_8 = h_SK_2_8.shape[1]
s = h_SK_2_32.shape[0] # Total kmers

print "#############  KMERS BASE 2 (16BITS) A BASE 2 (8 BITS)  #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B322B8.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B322B8 = programa.B322B8
B322B8.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32])

# Copy input data from host to device
d_SK_2_32 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_32)

# Output buffer
d_SK_2_8 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_8.nbytes)

# Execution Range
rango_global = (h_SK_2_8.shape[1], s)

# Kernel Execution
B322B8(cola, rango_global, None, d_SK_2_32, d_SK_2_8, cSK_2_32, cSK_2_8, s)

cola.finish()
# Retrieve output
cl.enqueue_copy(cola, h_SK_2_8, d_SK_2_8)

print h_SK_2_32
print h_SK_2_8
