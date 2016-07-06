import pyopencl as cl
import numpy as np
from time import time
import scipy
# BASE 10 A BASE 4

# Input Matrix
SK_2_16 = [[1,2,2, 255,255],[1,1,2,2,2],[2,7,7,3,3],[3,6,7,4,4], [2,1,1,4,4]]

h_SK_2_16 =np.ndarray((len(SK_2_16), len(SK_2_16[0]))).astype(np.uint16)
h_SK_2_16[:] = SK_2_16

cSK_2_16 = h_SK_2_16.shape[1]
rSK_2_16 = h_SK_2_16.shape[0]

rTMP = rSK_2_16

if (cSK_2_16%2)==0:
    cTMP = cSK_2_16
else:
    cTMP = (2  - cSK_2_16%2)  + cSK_2_16

# Output matrix
h_SK_2_32= np.ndarray((rTMP, cTMP/2)).astype(np.uint32)

print "#############  KMERS BASE 2 (8BITS) A BASE 2 (32 BITS)  #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B162B32.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B162B32 = programa.B162B32
B162B32.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32])

# Copy input data from host to device
d_SK_2_16 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_16)

# Temporal matrix , tamano de h_SK_2_16 x 2 dado que va a ser de 16 bits por elemento
size_TMP = (32 * rTMP * cTMP) / 8
d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)

# Output buffer
d_SK_2_32 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_32.nbytes)

# Execution Range
rango_global = (cTMP, rTMP)

cSK_2_32 = h_SK_2_32.shape[1]
# Kernel Execution
B162B32(cola, rango_global, None, d_SK_2_16, d_TMP, d_SK_2_32, cSK_2_16, cTMP, rTMP)
#B162B32(cola, (10,10), None, d_SK_2_16, d_SK_2_32, cSK_2_16, c_sk_2_32, s)

cola.finish()
# Retrieve output
cl.enqueue_copy(cola, h_SK_2_32, d_SK_2_32)
h_TMP = np.ndarray((rTMP, cTMP)).astype(np.uint32)
cl.enqueue_copy(cola, h_TMP, d_TMP)

print "h_TMP"
print "cTMP : " + str(cTMP) + " cSK_2_16: " + str(cSK_2_16)
print h_TMP
print h_SK_2_16
print h_SK_2_32
