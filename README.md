K-mersCL
========

Librerías OpenCL para el procesamiento paralelo masivo de K-mers en el ensamblaje genómico de-novo.

K-mersCL está basado en K-mass, que es un modelo de procesamiento paralelo masivo de K-mers de secuencias genómicas, basado en funciones definidas en términos de índices de espacios N-dimensionales, donde cada elemento del espacio representa un hilo de procesamiento que se encarga de calcular de forma concurrente cada uno de los posibles resultados de dichas funciones para un conjunto de datos definido.

K-mersCL se conforma por un conjunto de kernels OpenCL organizados en dos grupos:


Kernels para operaciones básicas
--------------------------------

* Obtención de los k-mers de una lectura (getKmers.py)
* Conversión de K-mers: Caracteres a sistema numérico base 4 (C2N.py)
* Conversión de K-mers: Sistema numérico base 4 a caracteres  (N2C.py)
* Conversión de K-mers: Sistema numérico base 4 al decimal (usando  enteros de 8 bits) (N2B8.py) y viceversa (B82N.py)
* Conversión de K-mers: Sistema numérico base 4 al decimal (usando  enteros de 16 bits) (N2B16.py) y viceversa (B162N.py)
* Conversión de K-mers: Sistema numérico base 4 al decimal (usando  enteros de 32 bits) (N2B32.py) y viceversa (B322N.py)
* Conversión de K-mers: Sistema numérico base 4 al decimal (usando  enteros de 64 bits) (N2B64.py) (D2N.py) y viceversa (B642N.py) (N2D.py)
* Conversión de K-mers: Sistema numérico decimal (usando  enteros de 8 bits) a numerico decimal (enteros de 16 bits) (B82B16.py) y viceversa (B162B8.py)
* Conversión de K-mers: Sistema numérico decimal (usando  enteros de 8 bits) a numerico decimal (enteros de 32 bits) (B82B32.py) y viceversa (B322B8.py)
* Conversión de K-mers: Sistema numérico decimal (usando  enteros de 8 bits) a numerico decimal (enteros de 64 bits) (B82B64.py) y viceversa (B642B8.py)
* Conversión de K-mers: Sistema numérico decimal (usando  enteros de 16 bits) a numerico decimal (enteros de 32 bits) (B162B32.py) y viceversa (B322B16.py)
* Conversión de K-mers: Sistema numérico decimal (usando  enteros de 16 bits) a numerico decimal (enteros de 64 bits) (B162B64.py) y viceversa (B642B16.py)
* Conversión de K-mers: Sistema numérico decimal (usando  enteros de 32 bits) a numerico decimal (enteros de 64 bits) (B322B64.py) y viceversa (B642B32.py)
* Inverso de K-mers (getR.py)
* Complemento de K-mers (getC.py)
* Complemento inverso de K-mers (getRC.py)
* K-mers Canónicos (getCanonical.py)

Kernels para operaciones relacionadas a estructuras de datos
------------------------------------------------------------

* Obtención de minimizers lexicográficos
* Obtención de frecuencias de posibles minimizers
* Obtención de minimizers por frecuencia
* Transformada Burrows-Wheeler
* Transformada inversa Burrows-Wheeler
* Función LF
* FM - index
* Función Occ
* Spookyhash3 (64bits)


*Nota:* Se requiere OpenCL 1.2 o superior
