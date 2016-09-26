__kernel void getMinimizers(
   __global uchar* SK_4, // Input matrix with base 4 representation of kmers
   __global uchar* ML_4, // Output matrix with minimizers
   __global uchar* ML_P, // Output matrix with positions of minimizers in k-mers
   __global uchar* TMP_P, // Output matrix with positions of minimizers in k-mers
   const unsigned int cSK_4, // Number of columns of SK_4
   const unsigned int cTMP_P, // Columns of TMP_P matriz
   const unsigned int m,  // M-mer size
   const unsigned int n_m,  // Number of M-mer
   const unsigned int s)  // Rows of SK_4 (Total kmers)
{

   int i = get_global_id(0);
   int j = get_global_id(1);

   //
   int pairs;
   // Iters for detect minimizer
   int iters = 0;


   // FILL TMP_P Matrix
   if((i <= cTMP_P-1)&& (j<=s-1))  {
     TMP_P[(j*cTMP_P)+i] = i;
  }

  float log_base = log2((float)cTMP_P);
  if ((int)log_base < log_base) {
     iters = (int)log_base + 1;
  } else {
     iters = (int) log_base;
  }

  int items = cTMP_P;
  for(int n=0; n<=iters; n++) {

      if (items%2==0) {
        pairs = items/2;
      } else {
        pairs = items/2 + 1;
      }

      if((i<=(pairs-1)) && (j<=s-1)) {

        ushort pos_A = (j*cSK_4) + (TMP_P[(j*cTMP_P)+(2*i*(1<<n))]);
        ushort pos_B = (j*cSK_4) + (TMP_P[(j*cTMP_P)+((2*i)+1)*(1<<n)]);

        if ((items%2!=0)&&(i==(pairs-1))) {
          TMP_P[(j*cTMP_P)+((i*2)*(1<<n))] = TMP_P[(j*cTMP_P)+((i*2)*(1<<n))];
        }
        else {
            for (int w=0; w<m; w++) {
                if (SK_4[pos_A+w] < SK_4[pos_B+w]) {
                  TMP_P[(j*cTMP_P)+((i*2)*(1<<n))] = TMP_P[(j*cTMP_P)+((i*2)*(1<<n))];
                  break;
                }
                else if (SK_4[pos_A+w] > SK_4[pos_B+w]) {
                  TMP_P[(j*cTMP_P)+((i*2)*(1<<n))] = TMP_P[(j*cTMP_P)+(((i*2)+1)*(1<<n))];
                  break;
                } else if (w==m-1) {
                  TMP_P[(j*cTMP_P)+((i*2)*(1<<n))] = TMP_P[(j*cTMP_P)+((i*2)*(1<<n))];
                  break;
                } else {  // Cuando funcione probar sin este else, que igual debe funcionar
                  continue;
                }

              }
        }


        }

          items = pairs;

  }

  // Extract first column from TMP_P Matrix (Contains positions)
  if((i == 0)&& (j<=s-1))  {
    ML_P[j] = TMP_P[(j*cTMP_P)];

 }

 // Relleno matriz con minimizers en base 4
  if((i <= m-1)&& (j<=s-1))  {
    ML_4[(j*m)+i] =  SK_4[(j*cSK_4)+i+ML_P[j]];
 }



}
