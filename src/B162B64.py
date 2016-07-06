# Este script define una matriz de kmers en base 4 y la transforma a una matriz de kmers en base 10 de 8 bits.
import pyopencl as cl
import numpy as np
from time import time
import scipy

# Input Matrix
SK_2_16 = [[1,1,1,2,6500,255],[1,2,6500,255,1,2],[2,1,300,100,7,3],[1,2,16,7,300,100], [3,1,1,4,300,100]]
h_SK_2_16 =np.ndarray((len(SK_2_16), len(SK_2_16[0]))).astype(np.uint16)
h_SK_2_16[:] = SK_2_16

cSK_2_16 = h_SK_2_16.shape[1]
rSK_2_16 = h_SK_2_16.shape[0]

rTMP = rSK_2_16

if (cSK_2_16%4)==0:
    cTMP = cSK_2_16
else:
    cTMP = (4  - cSK_2_16%4)  + cSK_2_16

# Output matrix
h_SK_2_64= np.ndarray((rTMP, cTMP/4)).astype(np.uint64)

print "#############  KMERS BASE 2 (8BITS) A BASE 2 (64 BITS)  #################"

# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B162B64.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B162B64 = programa.B162B64
B162B64.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32])

# Copy input data from host to device
d_SK_2_16 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_16)

# Temporal matrix , tamano de h_SK_2_16 x 2 dado que va a ser de 16 bits por elemento
size_TMP = (64 * rTMP * cTMP) / 8
d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)

# Output buffer
d_SK_2_64 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_64.nbytes)

# Execution Range
rango_global = (cTMP, rTMP)

cSK_2_64 = h_SK_2_64.shape[1]
# Kernel Execution
B162B64(cola, rango_global, None, d_SK_2_16, d_TMP, d_SK_2_64, cSK_2_16, cTMP, rTMP)
cola.finish()

# Retrieve output
cl.enqueue_copy(cola, h_SK_2_64, d_SK_2_64)
h_TMP = np.ndarray((rTMP, cTMP)).astype(np.uint64)
cl.enqueue_copy(cola, h_TMP, d_TMP)

print "h_TMP"
print "cTMP : " + str(cTMP) + " cSK_2_16: " + str(cSK_2_16)
print h_TMP
print h_SK_2_16
print h_SK_2_64
