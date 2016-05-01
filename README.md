K-mersCL
========

Librerías OpenCL para el procesamiento paralelo masivo de K-mers en el ensamblaje genómico de-novo.

K-mersCL está basado en K-mass, que es un modelo de procesamiento paralelo masivo de K-mers de secuencias genómicas, basado en funciones definidas en términos de índices de espacios N-dimensionales, donde cada elemento del espacio representa un hilo de procesamiento que se encarga de calcular de forma concurrente cada uno de los posibles resultados de dichas funciones para un conjunto de datos definido.

K-mersCL se conforma por un conjunto de kernels OpenCL organizados en dos grupos:


Kernels para operaciones básicas
--------------------------------

* Obtención de los k-mers de una lectura
* Conversión de K-mers: Caracteres a sistema numérico base 4
* Conversión de K-mers: Sistema numérico base 4 a caracteres
* Conversión de K-mers: Sistema numérico base 4 al decimal
* Conversión de K-mers: Sistema numérico decimal a base 4
* Inverso de K-mers
* Complemento de K-mers
* Complemento inverso de K-mers
* K-mers Canónicos

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
