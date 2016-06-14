__kernel void getRC(
   __global uchar* SK_4, // 8 bits (1 kmer por posicion) entrada // # DE COLUMNAS, DEBE SER MULTIPLO DE 4. SI K NO ES MULTIPLO DE 4, SE DEBE AGREGAR LAS COLUMNAS INICIALES IZQUIERDA= CON 0
   __global uchar* RSK_4, // 8 bits (4 kmers) salida, n cols = cSK_4 / 4
   __global uchar* RCSK_4, // 8 bits (4 kmers) salida, n cols = cSK_4 / 4
   const unsigned int cSK_4, // Numero de columnas de SK_4
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);


   //  Reverse
   if((i <= cSK_4-1)&& (j<=s-1))  {
     RSK_4[(j*cSK_4)+i] = SK_4[(j*cSK_4)+(cSK_4-i-1)];
    }

    // Complement
    if((i <= cSK_4-1)&& (j<=s-1))  {
      switch (RSK_4[(j*cSK_4)+i]) {
        case 0:
         RCSK_4[(j*cSK_4)+i] = 3;
         break;
        case 1:
         RCSK_4[(j*cSK_4)+i] = 2;
          break;
        case 2:
          RCSK_4[(j*cSK_4)+i] = 1;
          break;
        case 3:
          RCSK_4[(j*cSK_4)+i] = 0;
          break;
         default:
          RCSK_4[(j*cSK_4)+i] = 255;
           break;
      }
     }



  }
