import pyopencl as cl
import numpy as np
from time import time
# BASE 4 A CARACTERES
#Memoria para lectura en host
SK_4 = [[3,1,0,0,0,3,3,3,3,1,0,0,0,2,3,2,1,1],[3,1,0,0,0,2,3,2,1,1,0,0,0,2,3,2,1,1],[3,1,0,0,0,2,3,2,1,1,0,0,0,1,2,3,1,1]] # Matriz con KMERS
# SK_4 = [[3,3,3,3],[2,3,2,1],[1,2,3,1],[1,2,3,1],[0,1,2,3]]
h_SK_4 =np.ndarray((len(SK_4), len(SK_4[0]))).astype(np.uint8)
h_SK_4[:] = SK_4

rSK_4 = h_SK_4.shape[0] # Number of kmers
cSK_4 = h_SK_4.shape[1] # Kmer size

# TODO: Numpy, anadir columnas 0, tener el k de entrada de salida.
# Si cols de SK_4 no es multiplo de 4

rTMP = h_SK_4.shape[0]
if (h_SK_4.shape[1]%16)==0:
    cTMP =  cSK_4
else:
    cTMP = (16 - cSK_4%16) + cSK_4

# Output matrix
h_SK_2_32=np.ndarray((rTMP, cTMP/16)).astype(np.uint32)

print "#############  KMERS BASE 4 A BASE 2 (32bits) #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/N2B32.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
N2B32 = programa.N2B32
N2B32.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32])

#Buffer for intermedial matrix
# Num bits / 8
size_TMP = (32 * rTMP * cTMP) / 8
d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)

# Copy input data from host to device
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
# Output buffer
d_SK_2_32 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_32.nbytes)
# Execution Range
rango_global = (cTMP, rTMP)
# Kernel Execution
N2B32(cola, rango_global, None, d_SK_4, d_TMP, d_SK_2_32, cSK_4, cTMP, rTMP)
cola.finish()
# Retrieve output
cl.enqueue_copy(cola, h_SK_2_32, d_SK_2_32)
cola.flush()
h_TMP = np.ndarray(( rTMP, cTMP)).astype(np.uint32)
cl.enqueue_copy(cola, h_TMP, d_TMP)

print "cTMP: %s cSK_4: %s" % ( cTMP, cSK_4)
print "Matriz intermedia"
print h_TMP
print "Matriz entrada"
print h_SK_4
print "Matriz salida"
print h_SK_2_32
