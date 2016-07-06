import pyopencl as cl
import numpy as np
from time import time
import scipy

# Binary representation (4 bases 8 bits) to (16 bases 32 bits)

# Input Matrix
SK_2_8 = [[1,2,2, 255,255],[1,1,2,2,2],[2,7,7,3,3],[3,6,7,4,4], [2,1,1,4,4]]

h_SK_2_8 =np.ndarray((len(SK_2_8), len(SK_2_8[0]))).astype(np.uint8)
h_SK_2_8[:] = SK_2_8

cSK_2_8 = h_SK_2_8.shape[1]
rSK_2_8 = h_SK_2_8.shape[0]

rTMP = rSK_2_8

if (cSK_2_8%4)==0:
    cTMP = cSK_2_8
else:
    cTMP = (4  - cSK_2_8%4)  + cSK_2_8


# Output matrix
h_SK_2_32= np.ndarray((rTMP, cTMP/4)).astype(np.uint32)


print "#############  Binary representation (4 bases 8 bits) to (16 bases 32 bits) #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B82B32.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B82B32 = programa.B82B32
B82B32.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32])

# Copy input data from host to device
d_SK_2_8 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_8)

# Temporal matrix , tamano de h_SK_2_8 x 2 dado que va a ser de 16 bits por elemento
# Temporal matrix , tamano de h_SK_2_8 x 2 dado que va a ser de 16 bits por elemento
size_TMP = (32 * rTMP * cTMP) / 8
d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)

# Output buffer
d_SK_2_32 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_32.nbytes)

# Execution Range
rango_global = (cTMP, rTMP)
cSK_2_32 = h_SK_2_32.shape[1]

# Kernel Execution
B82B32(cola, rango_global, None, d_SK_2_8, d_TMP, d_SK_2_32, cSK_2_8, cTMP, rTMP)
cola.finish()

# Retrieve output
cl.enqueue_copy(cola, h_SK_2_32, d_SK_2_32)
h_TMP = np.ndarray((rTMP, cTMP)).astype(np.uint32)
cl.enqueue_copy(cola, h_TMP, d_TMP)

print "h_TMP"
print "cTMP : " + str(cTMP) + " cSK_2_8: " + str(cSK_2_8)
print h_TMP
print h_SK_2_8
print h_SK_2_32
