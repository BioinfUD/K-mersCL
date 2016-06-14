__kernel void N2B8(
   __global uchar* SK_4, // 8 bits (1 kmer por posicion) entrada // # DE COLUMNAS, DEBE SER MULTIPLO DE 4. SI K NO ES MULTIPLO DE 4, SE DEBE AGREGAR LAS COLUMNAS INICIALES IZQUIERDA= CON 0
   __global uchar* TMP, // Matriz intermedia donde se guardan corrimientos
   __global uchar* SK_2_8, // 8 bits (4 kmers) salida, n cols = cSK_4 / 4
   const unsigned int cSK_4, // Numero de columnas de SK_4
   const unsigned int cTMP, // Numero de columnas de SK_4
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);

   // Fill TMP matrix
   if((i <= cSK_4-1)&& (j<=s-1))  {
     TMP[j*cTMP + (i + cTMP - cSK_4)] = SK_4[(j*cSK_4)+i];
   }

   if((i < (cTMP-cSK_4))&& (j<= s-1)){
     TMP[j*cTMP + i] = 0;
   }

   // Left Shift
   if((i <= cTMP-1) && (i>=(cTMP-cSK_4)) && (j<=s-1))  {
     TMP[(j*cTMP)+i] = TMP[(j*cTMP)+i] << (8 - 2*((i%4)+1));
    }

    // Fussion
    for(int c=4; c>=2; c/=2) {
      if((i<=(cTMP-1)) && (j<=s-1)) {
        TMP[(j*cTMP)+i] =  TMP[(j*cTMP)+i*2] | TMP[(j*cTMP)+i*2+1];
      }
    }

    //Fill output matrix
    if((i <=((cTMP/4))-1)&& (j<=s-1))  {
      SK_2_8[(j*cTMP/4)+i] = TMP[(j*cTMP)+i];
    }

  }
