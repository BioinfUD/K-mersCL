__kernel void getCanonical(
   __global uchar* SK_4, // 8 bits (1 kmer por posicion) entrada // # DE COLUMNAS, DEBE SER MULTIPLO DE 4. SI K NO ES MULTIPLO DE 4, SE DEBE AGREGAR LAS COLUMNAS INICIALES IZQUIERDA= CON 0
   __global uchar* CA_I, // Storage for 64 bits decimal representation of reverse complement kmer matrix
   __global uchar* CASK_4, // Output matrix
   const unsigned int cSK_4, // Numero de columnas de SK_4
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);

   // REVERSE COMPLEMENT
   if((i <= cSK_4-1)&& (j<=s-1))  {
     CASK_4[(j*cSK_4)+i]=((SK_4[(j*cSK_4)+(cSK_4-1-i)])*3+3)%4;
    }

    // LEXICOGRAPH COMPARISSION
    if((i == 0)&& (j<=s-1)) {
      CA_I[j] = 0;
      for (int w=0; w<cSK_4; w++){
        if (CASK_4[(j*cSK_4)+w]<SK_4[(j*cSK_4)+w]) {
          CA_I[j] = 2;
          break;
        }
        else if (CASK_4[(j*cSK_4)+w]>SK_4[(j*cSK_4)+w]) {
          CA_I[j] = 1;
          break;
        }
      }
    }

    // WRITE OUTPUT matrix
    if ((i <= cSK_4-1)&& (j<=s-1)) {
      if (CA_I[j]==1) {
        CASK_4[(j*cSK_4)+i]=SK_4[(j*cSK_4)+i];
      }
    }

  }
