__kernel void getRC(
   __global uchar* SK_4, // 8 bits (1 kmer por posicion) entrada // # DE COLUMNAS, DEBE SER MULTIPLO DE 4. SI K NO ES MULTIPLO DE 4, SE DEBE AGREGAR LAS COLUMNAS INICIALES IZQUIERDA= CON 0
   __global uchar* RCSK_4, // 8 bits (4 kmers) salida, n cols = cSK_4 / 4
   const unsigned int k, // Numero de columnas de SK_4
   const unsigned int s)
{

   int x = get_global_id(0);
   int y = get_global_id(1);


   // REVERSE COMPLEMENT
   if((x <= k-1)&& (y<=s-1))  {
     RCSK_4[(y*k)+x]=((SK_4[(y*k)+(k-1-x)])*3+3)%4;
    }




  }
