import pyopencl as cl
import numpy as np
from time import time
import scipy
# Binary representation (4 bases 8 bits) to (32 bases 64 bits)

# Input Matrix
SK_2_8 = [[3,1,1,1,0,0,2,2, 255,255],[3,1,0,0,2,2,1,2,2,2],[3,1,1,2,2,2,7,7,3,3],[3,1,6,7,4,4,6,7,4,4], [3,3,6,7,4,4,1,1,4,4]]

h_SK_2_8 =np.ndarray((len(SK_2_8), len(SK_2_8[0]))).astype(np.uint8)
h_SK_2_8[:] = SK_2_8

cSK_2_8 = h_SK_2_8.shape[1]
rSK_2_8 = h_SK_2_8.shape[0]

rTMP = rSK_2_8

if (cSK_2_8%8)==0:
    cTMP = cSK_2_8
else:
    cTMP = (8  - cSK_2_8%8)  + cSK_2_8

# Output matrix
h_SK_2_64= np.ndarray((rTMP, cTMP/8)).astype(np.uint64)

print "#############  Binary representation (4 bases 8 bits) to (32 bases 64 bits)  #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B82B64.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B82B64 = programa.B82B64
B82B64.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32])

# Copy input data from host to device
d_SK_2_8 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_8)

# Temporal matrix , tamano de h_SK_2_8 x 2 dado que va a ser de 16 bits por elemento
size_TMP = (64 * rTMP * cTMP) / 8
d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)

# Output buffer
d_SK_2_64 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_64.nbytes)

# Execution Range
rango_global = (cTMP, rTMP)
cSK_2_64 = h_SK_2_64.shape[1]

# Kernel Execution
B82B64(cola, rango_global, None, d_SK_2_8, d_TMP, d_SK_2_64, cSK_2_8, cTMP, rTMP)
cola.finish()

# Retrieve output
cl.enqueue_copy(cola, h_SK_2_64, d_SK_2_64)
h_TMP = np.ndarray((rTMP, cTMP)).astype(np.uint64)
cl.enqueue_copy(cola, h_TMP, d_TMP)

print "h_TMP"
print "cTMP : " + str(cTMP) + " cSK_2_8: " + str(cSK_2_8)
print h_TMP
print h_SK_2_8
print h_SK_2_64
