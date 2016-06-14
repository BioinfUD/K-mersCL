# Get kmers complement using heterogeneous computing
import pyopencl as cl
import numpy as np
from time import time

#SK_4 = [[3,3,1,3],[2,3,2,1],[1,2,3,1],[1,2,3,1],[0,1,2,3],[2,2,3,3]] # Matriz con KMERS
SK_4 = [
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    [3,0,0,0,3,3,3,3,1,0,0,0,3,3,3,3,1,0,0,0,3,3,3,3,1,0,0,0,3,3,3,3,1],
    [1,0,0,0,1,2,3,1,1,0,0,0,1,2,3,1,1,0,0,0,1,2,3,1,1,0,0,0,1,2,3,1,1]
    ]
h_SK_4 =np.ndarray((len(SK_4), len(SK_4[0]))).astype(np.uint8)
h_SK_4[:] = SK_4

S = h_SK_4.shape[0] # Numeero de kmes
K = h_SK_4.shape[1] # Tamano del kmer
cSK_4 = h_SK_4.shape[1]

# Matriz de salida de complemento
h_CSK_4 = np.ndarray(h_SK_4.shape).astype(np.uint8)
# Matriz de complemento reverso
h_RCSK_4 = np.ndarray(h_SK_4.shape).astype(np.uint8)



rTMP = h_SK_4.shape[0]
if (h_SK_4.shape[1]%32)==0:
    cTMP =  cSK_4
else:
    cTMP = (32 - cSK_4%32) + cSK_4

print "Get canonical"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)

# Program for reverse complement
codigo_kernel_c = open("kernels/getCanonical.cl").read()
programa_c = cl.Program(contexto, codigo_kernel_c).build()
getCanonical = programa_c.getCanonical
getCanonical.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None, np.uint32, np.uint32, np.uint32])
# NDRange
rango_global = (cTMP, S)

#Copio datos de entrada a dispositivo
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
#Complement buffer
d_RSK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_CSK_4.nbytes)
# Reverse complement buffer
d_RCSK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_RCSK_4.nbytes)
# TMPs Buffers
size_TMP = (64 * rTMP * cTMP) / 8
d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)
d_TMPRC = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)
# Buffers for decimal representation
h_RCSK_10 =np.ndarray((rTMP, cTMP/32)).astype(np.uint64)
h_SK_10 =np.ndarray((rTMP,cTMP/32)).astype(np.uint64)
d_SK_10 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_RCSK_10.nbytes)
d_RCSK_10 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_RCSK_10.nbytes)
# Buffer for output matrix
d_CASK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_4.nbytes)
h_CASK_4 =np.ndarray((S, K)).astype(np.uint8)
# Kernel execution for canonical kmer
getCanonical(cola, rango_global, None, d_SK_4, d_RSK_4, d_RCSK_4, d_TMP, d_TMPRC, d_SK_10, d_RCSK_10, d_CASK_4, cSK_4, cTMP, S)
cola.finish()
# Traigo datos

h_TMP =np.ndarray((rTMP, cTMP)).astype(np.uint64)
h_TMPRC =np.ndarray((rTMP, cTMP)).astype(np.uint64)
h_RCSK_4 =np.ndarray((rTMP, cSK_4)).astype(np.uint8)
cl.enqueue_copy(cola, h_RCSK_4, d_RCSK_4)
cl.enqueue_copy(cola, h_TMP, d_TMP)
cl.enqueue_copy(cola, h_TMPRC, d_TMPRC)
cl.enqueue_copy(cola, h_CASK_4, d_CASK_4)
cl.enqueue_copy(cola, h_RCSK_10, d_RCSK_10)
cl.enqueue_copy(cola, h_SK_10, d_SK_10)



print "Columns of TMP matrix {}, Columns of input matrix {}, Number of colums of".format(cTMP, cSK_4 )
print "Input matrix"
print h_SK_4
print "RCSK_4"
print h_RCSK_4
print "Decimal Original"
print h_SK_10
print "Decimal Reverse Complement"
print h_RCSK_10
print "Output matrix"
print h_CASK_4
