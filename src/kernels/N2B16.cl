__kernel void N2B16(
   __global uchar* SK_4, // 8 bits (1 nucleotide per cell)
   __global ushort* TMP, // Temporal matrix
   __global ushort* SK_2_16, // 16 bits (8 bases)
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
     TMP[(j*cTMP)+i] = TMP[(j*cTMP)+i] << (16 - 2*((i%8)+1));
    }

    // Fussion
    for(int c=8; c>=2; c/=2) {
      if((i<=(cTMP-1)) && (j<=s-1)) {
        TMP[(j*cTMP)+i] = TMP[(j*cTMP)+i*2] | TMP[(j*cTMP)+i*2+1];
      }
    }

    //Fill output matrix
    if((i <=((cTMP/8))-1)&& (j<=s-1))  {
      SK_2_16[(j*cTMP/8)+i] = TMP[(j*cTMP)+i];
    }

  }
