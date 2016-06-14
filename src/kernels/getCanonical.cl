__kernel void getCanonical(
   __global uchar* SK_4, // 8 bits (1 kmer por posicion) entrada // # DE COLUMNAS, DEBE SER MULTIPLO DE 4. SI K NO ES MULTIPLO DE 4, SE DEBE AGREGAR LAS COLUMNAS INICIALES IZQUIERDA= CON 0
   __global uchar* RSK_4, // 8 bits (4 kmers) salida, n cols = cSK_4 / 4
   __global uchar* RCSK_4, // 8 bits (4 kmers) salida, n cols = cSK_4 / 4
   __global ulong* TMP, // Storage for intermedial 64 bits decimal representation of original kmer matrix
   __global ulong* TMPRC, // Storage for intermedial 64 bits decimal representation of reverse complement kmer matrix
   __global ulong* SK_10, // Storage for 64 bits decimal representation of reverse complement kmer matrix
   __global ulong* RCSK_10, // Storage for 64 bits decimal representation of reverse complement kmer matrix
   __global uchar* CASK_4, // Output matrix
   const unsigned int cSK_4, // Numero de columnas de SK_4
   const unsigned int cTMP, // Numero de columnas de SK_4
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);

   // REVERSE COMPLEMENT
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


     // DECIMAL CONVERSION FOR ORIGINAL AND REVERSE COMPLEMENT
     // Fill TMP matrix
     if((i <= cSK_4-1)&& (j<=s-1))  {
       TMP[j*cTMP + (i + cTMP - cSK_4)] = SK_4[(j*cSK_4)+i];
     }

     if((i < (cTMP-cSK_4))&& (j<= s-1)){
       TMP[j*cTMP + i] = 0;
     }

     // Left Shift
     if((i <= cTMP-1) && (i>=(cTMP-cSK_4)) && (j<=s-1))  {
       TMP[(j*cTMP)+i] = (ulong) TMP[(j*cTMP)+i] << (64 - 2*((i%32)+1));

      }

      // Fussion
      for(int c=32; c>=2; c/=2) {
        if((i<=(cTMP-1)) && (j<=s-1)) {
          TMP[(j*cTMP)+i] = (ulong) TMP[(j*cTMP)+i*2] | TMP[(j*cTMP)+i*2+1];
        }
      }

      //Fill output matrix
      if((i <=((cTMP/32))-1)&& (j<=s-1))  {
        SK_10[(j*cTMP/32)+i] = (ulong) TMP[(j*cTMP)+i];
      }

      // Fill TMP matrix
      if((i <= cSK_4-1)&& (j<=s-1))  {
        TMPRC[j*cTMP + (i + cTMP - cSK_4)] = RCSK_4[(j*cSK_4)+i];
      }

      if((i < (cTMP-cSK_4))&& (j<= s-1)){
        TMPRC[j*cTMP + i] = 0;
      }

      // Left Shift
      if((i <= cTMP-1) && (i>=(cTMP-cSK_4)) && (j<=s-1))  {
        TMPRC[(j*cTMP)+i] = (ulong) TMPRC[(j*cTMP)+i] << (64 - 2*((i%32)+1));

       }

       // Fussion
       for(int c=32; c>=2; c/=2) {
         if((i<=(cTMP-1)) && (j<=s-1)) {
           TMPRC[(j*cTMP)+i] = (ulong) TMPRC[(j*cTMP)+i*2] | TMPRC[(j*cTMP)+i*2+1];
         }
       }

       //Fill output matrix
       if((i <=((cTMP/32))-1)&& (j<=s-1))  {
         RCSK_10[(j*cTMP/32)+i] = (ulong) TMPRC[(j*cTMP)+i];
       }


      //COMPARISSION

      if (cSK_4<=32) {
        if((i <= cSK_4-1)&& (j<=s-1))  {
          if(SK_10[(j*cTMP/32)]<RCSK_10[(j*cTMP/32)]) { // Unique position
            CASK_4[j*cSK_4 + i] = SK_4[j*cSK_4 + i] ;
          } else {
            CASK_4[j*cSK_4 + i] = RCSK_4[j*cSK_4 + i] ;
          }
        }
      }

      if (cSK_4>32 && cSK_4<=64) {
        if((i <= cSK_4-1)&& (j<=s-1))  {
          if(SK_10[(j*cTMP/32)]<RCSK_10[(j*cTMP/32)]) { // Unique position
            CASK_4[j*cSK_4 + i] = SK_4[j*cSK_4 + i] ;
          } else if (SK_10[(j*cTMP/32)]>RCSK_10[(j*cTMP/32)]) {
            CASK_4[j*cSK_4 + i] = RCSK_4[j*cSK_4 + i] ;
          } else {
            if(SK_10[(j*cTMP/32)+1]<RCSK_10[(j*cTMP/32)+1]) { // Unique position
              CASK_4[j*cSK_4 + i] = SK_4[j*cSK_4 + i] ;
            } else {
              CASK_4[j*cSK_4 + i] = RCSK_4[j*cSK_4 + i] ;
            }
          }
        }
      }




  }
