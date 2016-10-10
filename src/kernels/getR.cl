__kernel void getR(
   __global uchar* SK_4, // 8 bits (1 kmer por posicion) entrada // # DE COLUMNAS, DEBE SER MULTIPLO DE 4. SI K NO ES MULTIPLO DE 4, SE DEBE AGREGAR LAS COLUMNAS INICIALES IZQUIERDA= CON 0
   __global uchar* RSK_4, // 8 bits (4 kmers) salida, n cols = cSK_4 / 4
   const unsigned int cSK_4, // Numero de columnas de SK_4
   const unsigned int s)
{

   int x = get_global_id(0);
   int y = get_global_id(1);


   if((x <= cSK_4-1)&& (y<=s-1))  {
     RSK_4[(y*cSK_4)+x] = SK_4[(y*cSK_4)+(cSK_4-x-1)];
    }


  }
