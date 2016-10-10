#pragma OPENCL EXTENSION cl_khr_global_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics:  enable
#pragma OPENCL EXTENSION cl_khr_local_int64_base_atomics : enable

__kernel void getCanonical(
   __global uchar* SK_4, // 8 bits (1 kmer por posicion) entrada // # DE COLUMNAS, DEBE SER MULTIPLO DE 4. SI K NO ES MULTIPLO DE 4, SE DEBE AGREGAR LAS COLUMNAS INICIALES IZQUIERDA= CON 0
   __global uchar* CA_I, // Storage for 64 bits decimal representation of reverse complement kmer matrix
   __global uchar* CASK_4, // Output matrix
   __global ulong* SK_10_64, // Output matrix
   __global ulong* CASK_10_64, // Output matrix
   const unsigned int k, // Numero de columnas de SK_4
   const unsigned int s,
   const unsigned int cSK_10_64)
{

  int x = get_global_id(0);
  int y = get_global_id(1);

  // FUNCTION: Reverse complement
  if((x <= k-1)&& (y<=s-1))  {
    CASK_4[(y*k)+x]=((SK_4[(y*k)+(k-1-x)])*3+3)%4;
   }


   // Initialize output matrix
    if((x <= k-1)&& (y<=s-1))  {
      SK_10_64[(y*cSK_10_64)+x]=0;
      CASK_10_64[(y*cSK_10_64)+x]=0;
     }

    // FUNCTION: Get decimal value

    if ((x<k) && (y<s))  {
      ulong n1 = (ulong) SK_4[y*k + x]<<((k-x-1)%32)*2;
      atomic_or(&SK_10_64[(y*cSK_10_64)  + ((k-1)/32-(k-x-1)/32)], n1 );
      ulong n2 = (ulong) CASK_4[y*k + x]<<((k-x-1)%32)*2;
      atomic_or(&CASK_10_64[(y*cSK_10_64)  + ((k-1)/32-(k-x-1)/32)], n2 );
    }
    barrier(CLK_GLOBAL_MEM_FENCE);


    // FUNCTION: Lexicographical comparission
    if ((x<k) && (y<s))  {
        if (CASK_10_64[y*cSK_10_64]<SK_10_64[y*cSK_10_64])
          {
            CA_I[y]=2;
          }
        else if (CASK_10_64[y*cSK_10_64]>SK_10_64[y*cSK_10_64])
          {
            CA_I[y]=1;
          }
        else if((CASK_10_64[y*cSK_10_64]==SK_10_64[y*cSK_10_64]) && (k<32))
          {
            CA_I[y]=0;
          }
        else
          {
          if (CASK_10_64[y*cSK_10_64+1]<SK_10_64[y*cSK_10_64+1])
            {
              CA_I[y]=2;
            }
          else if (CASK_10_64[y*cSK_10_64+1]>SK_10_64[y*cSK_10_64+1])
            {
              CA_I[y]=1;
            }
          else
            {
              CA_I[y]=0;
            }
          }
    }


    // Function: Update output matrix
    if ((x <= k-1)&& (y<=s-1)) {
      if (CA_I[y]==1) {
        CASK_4[(y*k)+x]=SK_4[(y*k)+x];
      }
    }



  }
