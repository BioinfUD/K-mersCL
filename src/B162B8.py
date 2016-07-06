# Este script define una matriz de kmers en base 4 y la transforma a una matriz de kmers en base 10 de 8 bits.
import pyopencl as cl
import numpy as np
from time import time
import scipy

# BASE 10 A BASE 4

# Input Matrix
SK_2_16 = [[256,2],[512,2],[65535,1]]
h_SK_2_16 =np.ndarray((len(SK_2_16), len(SK_2_16[0]))).astype(np.uint16)
h_SK_2_16[:] = SK_2_16


#Output matrix
h_SK_2_8 =np.ndarray((len(SK_2_16), len(SK_2_16[0])*2)).astype(np.uint8)

print h_SK_2_8
cSK_2_16 = h_SK_2_16.shape[1]
cSK_2_8 = h_SK_2_8.shape[1]
s = h_SK_2_16.shape[0] # Total kmers


print "#############  KMERS BASE 2 (16BITS) A BASE 2 (8 BITS)  #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B162B8.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B162B8 = programa.B162B8
B162B8.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32])

# Copy input data from host to device
d_SK_2_16 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_16)

# Output buffer
d_SK_2_8 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_8.nbytes)

# Execution Range
rango_global = (h_SK_2_8.shape[1], s)

# Kernel Execution
B162B8(cola, rango_global, None, d_SK_2_16, d_SK_2_8, cSK_2_16, cSK_2_8, s)

cola.finish()
# Retrieve output
cl.enqueue_copy(cola, h_SK_2_8, d_SK_2_8)


print h_SK_2_16
print h_SK_2_8
