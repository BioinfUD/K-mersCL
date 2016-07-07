__kernel void B322B64(
   __global uint* SK_2_32, // 8 bits (1 kmer por posicion) entrada // # DE COLUMNAS, DEBE SER MULTIPLO DE 4. SI K NO ES MULTIPLO DE 4, SE DEBE AGREGAR LAS COLUMNAS INICIALES IZQUIERDA= CON 0
   __global ulong* TMP, // Matriz intermedia donde se guardan corrimientos
   __global ulong* SK_2_64, // 16 bits (8 kmers) salida, n cols = cSK_4 / 4
   const unsigned int cSK_16, // Numero de columnas de SK_4
   const unsigned int cTMP, // Numero de columnas de SK_4
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);

   // Fill TMP matrix
   if((i <= cSK_16-1)&& (j<=s-1))  {
     TMP[j*cTMP + (i + cTMP - cSK_16)] = SK_2_32[(j*cSK_16)+i];
   }

   if((i < (cTMP-cSK_16))&& (j<= s-1)){
     TMP[j*cTMP + i] = 0;
   }

   // Left Shift
   if((i <= cTMP-1) && (i>=(cTMP-cSK_16)) && (j<=s-1))  {
     TMP[(j*cTMP)+i] = (ulong) TMP[(j*cTMP)+i] << (64 - 32*((i%2)+1));
    }

    // Fussion
    for(int c=2; c>=2; c/=2) {
      if((i<=(cTMP-1)) && (j<=s-1)) {
        TMP[(j*cTMP)+i] = (ulong)  TMP[(j*cTMP)+i*2] | TMP[(j*cTMP)+i*2+1];
      }
    }

     // Fill output matrix
    if((i <=((cTMP/2))-1)&& (j<=s-1))  {
      SK_2_64[(j*cTMP/2)+i] =  (ulong) TMP[(j*cTMP)+i];
    }
  }
