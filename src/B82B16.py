# Este script define una matriz de kmers en base 4 y la transforma a una matriz de kmers en base 10 de 8 bits.
import pyopencl as cl
import numpy as np
from time import time
import scipy
# BASE 10 A BASE 4

# Matriz de entrada
#K_2_8 = [[255,255,1,1,1,1,1,1,1,2]]

#SK_2_8 = [[254,255],[254,255]]
#SK_2_8 = [[255,255],[2,2],[3,3],[4,4], [4,4]]
SK_2_8 = [[1,3,1,1,255,255],[1,3,2,1,2,2],[1,3,3,1,3,3],[1,3,2,1,4,4], [1,3,2,1,4,4]]

h_SK_2_8 =np.ndarray((len(SK_2_8), len(SK_2_8[0]))).astype(np.uint8)
h_SK_2_8[:] = SK_2_8

cSK_2_8 = h_SK_2_8.shape[1]
rSK_2_8 = h_SK_2_8.shape[0]

rTMP = rSK_2_8

if (cSK_2_8%2)==0:
    cTMP = cSK_2_8
else:
    cTMP = cSK_2_8  + 1


# Matriz de salida
h_SK_2_16= np.ndarray((rTMP, cTMP/2)).astype(np.uint16)


print "#############  KMERS BASE 2 (8BITS) A BASE 2 (16 BITS)  #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B82B16.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B82B16 = programa.B82B16
B82B16.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32])

#Copio datos de entrada a dispositivo
d_SK_2_8 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_8)

#Matriz temporal , tamano de h_SK_2_8 x 2 dado que va a ser de 16 bits por elemento
size_TMP = (16 * rTMP * cTMP) / 8
d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)

#Bufer para la salida
d_SK_2_16 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_16.nbytes)

# Dimensiones de ejecucion
rango_global = (cTMP, rTMP)

cSK_2_16 = h_SK_2_16.shape[1]
# Ejecucion del kernel
B82B16(cola, rango_global, None, d_SK_2_8, d_TMP, d_SK_2_16, cSK_2_8, cTMP, rTMP)
#B82B16(cola, (10,10), None, d_SK_2_8, d_SK_2_16, cSK_2_8, cSK_2_16, s)

cola.finish()
# Traigo datos
cl.enqueue_copy(cola, h_SK_2_16, d_SK_2_16)
h_TMP = np.ndarray((rTMP, cTMP)).astype(np.uint16)
cl.enqueue_copy(cola, h_TMP, d_TMP)

print "h_TMP"
print "cTMP : " + str(cTMP) + " cSK_2_8: " + str(cSK_2_8)
print h_TMP
print h_SK_2_8
print h_SK_2_16
