__kernel void B82B32(
   __global uchar* SK_2_8, // 8 bits (1 kmer por posicion) entrada // # DE COLUMNAS, DEBE SER MULTIPLO DE 4. SI K NO ES MULTIPLO DE 4, SE DEBE AGREGAR LAS COLUMNAS INICIALES IZQUIERDA= CON 0
   __global uint* TMP, // Matriz intermedia donde se guardan corrimientos
   __global uint* SK_2_32, // 16 bits (8 kmers) salida, n cols = cSK_4 / 4
   const unsigned int cSK_8, // Numero de columnas de SK_4
   const unsigned int cTMP, // Numero de columnas de SK_4
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);

   // Fill TMP matrix
   if((i <= cSK_8-1)&& (j<=s-1))  {
     TMP[j*cTMP + (i + cTMP - cSK_8)] = SK_2_8[(j*cSK_8)+i];
   }

   if((i < (cTMP-cSK_8))&& (j<= s-1)){
     TMP[j*cTMP + i] = 0;
   }

   // Left Shift
   if((i <= cTMP-1) && (i>=(cTMP-cSK_8)) && (j<=s-1))  {
     TMP[(j*cTMP)+i] = TMP[(j*cTMP)+i] << (32 - 8*((i%4)+1));
    }

    // Fussion
    // N iteraciones for: 1, log_2(n/m), 8n es destino, 8m
    for(int c=4; c>=2; c/=2) {
      if((i<=(cTMP-1)) && (j<=s-1)) {
        TMP[(j*cTMP)+i] =  TMP[(j*cTMP)+i*2] | TMP[(j*cTMP)+i*2+1];
      }
    }

     // Fill output matrix
    if((i <=((cTMP/4))-1)&& (j<=s-1))  {
      SK_2_32[(j*cTMP/4)+i] = TMP[(j*cTMP)+i];
    }
  }
