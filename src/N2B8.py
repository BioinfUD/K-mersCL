# Este script define una matriz de kmers en base 4 y la transforma a una matriz de kmers en base 10 de 8 bits.
import pyopencl as cl
import numpy as np
from time import time
# BASE 4 A CARACTERES
#Memoria para lectura en host
SK_4 = [[1,0,3,3,3,3,1],[1,0,2,3,2,1,1],[2,0,1,2,3,1,1]] # Matriz con KMERS
#SK_4 = [[3,3,3,3],[2,3,2,1],[1,2,3,1],[1,2,3,1],[0,1,2,3]] # Matriz con KMERS
h_SK_4 =np.ndarray((len(SK_4), len(SK_4[0]))).astype(np.uint8)
h_SK_4[:] = SK_4

rSK_4 = h_SK_4.shape[0] # Number of kmers
cSK_4 = h_SK_4.shape[1] # Kmer size

# TODO: Numpy, anadir columnas 0, tener el k de entrada de salida.
# Si cols de SK_4 no es multiplo de 4

rTMP = h_SK_4.shape[0]
if (h_SK_4.shape[1]%4)==0:
    cTMP =  cSK_4
else:
    cTMP = (4 - cSK_4%4) + cSK_4


# Output matrix
h_SK_2_8=np.ndarray((rTMP, cTMP/4)).astype(np.uint8)

print "#############  KMERS BASE 4 A BASE 2 (8bits) #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/N2B8.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
N2B8 = programa.N2B8
N2B8.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32])

#Buffer for intermedial matrix
# Num bits / 8
size_TMP = (8 * rTMP * cTMP) / 8
d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)

#Copio datos de entrada a dispositivo
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
#Bufer para la salida
d_SK_2_8 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_8.nbytes)
# Dimensiones de ejecucion
rango_global = (cTMP, rTMP)
# Ejecucion del kernel
N2B8(cola, rango_global, None, d_SK_4, d_TMP, d_SK_2_8, cSK_4, cTMP, rTMP)
cola.finish()
# Traigo datos
cl.enqueue_copy(cola, h_SK_2_8, d_SK_2_8)
cola.flush()

print "cTMP: %s cSK_4: %s" % ( cTMP, cSK_4)
print "Matriz entrada"
print h_SK_4
print "Matriz salida"
print h_SK_2_8
