__kernel void N2B32(
   __global uchar* SK_4, // 8 bits (1 nucleotide per cell)
   __global uint* TMP, // Temporal matrix
   __global uint* SK_2_32, // 32 bits (32 bits per 16 bases)
   const unsigned int cSK_4, // Number of columns of input matrix
   const unsigned int cTMP, // Number of columns of TMP matrix
   const unsigned int s) // Number of kmers
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
     TMP[(j*cTMP)+i] = TMP[(j*cTMP)+i] << (32 - 2*((i%16)+1));
    }

    // Fussion
    for(int c=16; c>=2; c/=2) {
      if((i<=(cTMP-1)) && (j<=s-1)) {
        TMP[(j*cTMP)+i] = TMP[(j*cTMP)+i*2] | TMP[(j*cTMP)+i*2+1];
      }
    }

    //Fill output matrix
    if((i <=((cTMP/16))-1)&& (j<=s-1))  {
      SK_2_32[(j*cTMP/16)+i] = TMP[(j*cTMP)+i];
    }

  }
