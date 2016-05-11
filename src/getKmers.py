#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyopencl as cl
import numpy as np
from time import time

#Memoria para lectura en host
R = "TCAGCTACGTCAGCTACAGTCA"
R_h = np.chararray((len(R)))
R_h [:] = list(R)
L = len(R_h)
K = 3

############ EXTRAER KMERS ##################
print "############ EXTRAER KMERS ##################"
# OpenCL things
codigo_kernel = open("kernels/getKR.cl").read()
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
programa = cl.Program(contexto, codigo_kernel).build()
#Memoria para salida
numero_celdas = ((L-K+1)*K)
KR_h = np.chararray((numero_celdas))
#Memoria en dispositivo
#Copio R_h y Apartor espacio para KR_d
R_d =  cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=R_h)
KR_d = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, (L-K+1)*K )
#Dimensiones
rango_global = (K, L-K+1)
#Ejecucion del kernel
getKR = programa.getKR
getKR.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])
rtime = time()
getKR(cola, rango_global, None, R_d, KR_d, K, L)
cola.finish()
rtime = time() - rtime
print "El kernel se ejecutó en ", rtime, "segundos"
#Traigo datos
cl.enqueue_copy(cola, KR_h, KR_d)
print R_h
print KR_h
