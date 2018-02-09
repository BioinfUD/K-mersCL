// All divisions used here are integer division
// Variables inside a kernel not declared with an address space qualifier are considered Private

__kernel void getSuperK_M(
   __global uint* RSK_G, // Vector of reads and super k-mers (32 bits)
   __global uint* counters, // Matrix of temporal output
   const unsigned int nr, // Number of reads
   const unsigned int r, // Read size
   const unsigned int k, // K-mer size
   const unsigned int m // m-mer size
   )
{
   uint x, y, yl, xl, p, start, offset, nmk, nm, min, a, b, c, d, e, idt, ts, nt, lsd, nkr, tmp, ls, start_zone, start_superk, nt1, nt2, nt3, nt4 = 0;
   bool f, flag, g;

   x = get_global_id(0);
   y = get_global_id(1);
   xl = get_local_id(0);
   yl = get_local_id(1);

   __local uint RSK[READ_SIZE]; // Vector to store reads, m-mers and superk-mers. This vector is overwritten in each stage, size: length of the read
   __local uint RCMT[SECOND_MT]; // Reverse complement of current m-mer of tile (32 bits), size: nt2
   __local uint MT[NT_MAX]; // Current m-mer of tile (32 bits), size: Greater between nt2 and nt3
   __local uint counter; // Number of super k-mers found (32 bits)
   __local uint mp; //Current canonical minimizer or signature position
   __local uint nmp; //New minimizer position.
   __local uint minimizer;  // Current canonical minimizer or signature

   nmk = k - m + 1;  //Number of m-mers per k-mers

   // PROCESS: Global to local
   // Reads in global memory to Read in local memory

   ts = get_local_size(0); // ts: Tile size, work group size
   nt1 = (r-1)/ts + 1; // nt1: Number of tiles for this process

   if ((x<ts) && (x<r) && (y<nr)){
      for (int i=0; i<nt1; i++) { // coalesced
        p=ts*i+x;
        if (p > r-1) {
          break;
        }
        RSK[p]=RSK_G[y*r+p];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // PROCESS: getCm-mers
    // Reads to decimal representation of Canonical M-mers

    lsd = get_local_size(0) / m; // lsd: Local space divisions
    nm = r - m + 1; // nm: Number of m-mers per read
    ts = ((nm-1)/lsd) + 1;  // Tile size for this process
    nt2 = ((nm-1)/ts) + 1 ; // Number of tiles or number of sub-reads per read
    a = A_MASK; // a:Mask (Sample: 63 for m = 4, to cancel all bits different to first 6)


    // Parallel computing of the first m-mer of each tile
    if ((x < m*nt2) && (y<nr)) {
      idt = x/m;  // Tile id
      start = ts*idt; // Position of the first base for each tile
      offset = x % m;
      p = (start) + (offset);  // Position of the base corresponding a the thread
      MT[idt] = 0; // MT Reset
      RCMT[idt] = 0; // RCMT Reset
      b = RSK[p]; // Each thread copies its base to b
      atomic_or(&MT[idt], (b << (m-(x%m)-1)*2)); //Calculating the first m-mer of each tile
      atomic_or(&RCMT[idt], (((~b) & 3) << (x%m)*2)); //Calculating the reverse complement of the first m-mer of each tile
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Checking for Signature
    
    if ((x < nt2) && (y<nr)) {
      idt = x;
      start = ts*idt;
      f = false;
      g = false;
      c = MT[idt];
      e = (c >> (m-3)*2) & 0x0000003F;
      if ((e==0) || (e==4)) {
        f = true;
      }
      else {
        for (int w=0; w<m-2; w++) {
          e = ((c >> (w*2)) & 0x0000000F);
          if (e==0) {
            f = true;
          }
        }

      }
      d = RCMT[idt];
      e = (d >> ((m-3)*2)) & 0x0000003F;
      if ((e == 0) || (e == 4)) {
        //printf("G assigment first if \n");
        g = true;
      }
      else {
        for (int w=0; w<(m-2); w++) {
          e = ((d >> (w*2)) & 0x0000000F);
          if (e==0) {
            g = true;
          }
        }

      }
      if (!f && !g) {
        RSK[start + m -1] = (c < d) ? c : d;
      }
      else if (f && !g) {
        //printf("F = true, G=false, d=%d, e=%d, x=%d, start + m -1 = %d",d, e ,x, start + m -1);
        RSK[start + m -1] = d;
      }
      else if (!f && g) {
        RSK[start + m -1] = c;
      } else {
        RSK[start + m -1] = 1 << (m*2);
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Serial computation of the m-mers remaining of each tile
    
    if ((x<nt2) && (y<nr))	{
       idt = x; // Tile id
       start = ts*idt; // Position of the first base for each tile
       c = MT[idt]; // Copy to c the decimal value of the first m-mer of each tile
       d = RCMT[idt]; // Copy to d the decimal value of the reverse complement of the first m-mer of each tile

       for (int z=0; z<(ts-1); z++) {
       		if (start+m+z > r-1) {
            break;
          }
           b = RSK[start+m+z]; // Copy to b the last base of the m-mer that will be calculated (current m-mer)
           c = ((c & a) << 2)|b; // Calculating the current m-mer from the previous m-mer and the new base read (b)
           d = ((d>>2) & (a)) | (((~b) & 3)<<((m-1)*2)); // Calculating the reverse complement of the current m-mer from the reverse complement of the previous m-mer and the new base read (b)
            
           // Checking for Signature

           f = false;
           g = false;

           e = (c >> (m-3)*2) & 0x0000003F;
           if ((e==0) || (e==4)) {
             f = true;
           }
           else {
             for (int w=0; w<m-2; w++) {
               e = ((c >> (w*2)) & 0x0000000F);
               if (e==0) {
                 f = true;
               }
             }

           }
           e = (d >> ((m-3)*2)) & 0x0000003F;
           if ((e == 0) || (e == 4)) {
             //printf("G assigment first if \n");
             g = true;
           }
           else {
             for (int w=0; w<(m-2); w++) {
               e = ((d >> (w*2)) & 0x0000000F);
               if (e==0) {
                 g = true;
               }
             }
           }
           if (!f && !g) {
             RSK[start + m +z] = (c < d) ? c : d;
           }
           else if (f && !g) {
             //printf("F = true, G=false, d=%d, e=%d, x=%d, start + m -1 = %d",d, e ,x, start + m -1);
             RSK[start + m +z] = d;
           }
           else if (!f && g) {
             RSK[start + m +z] = c;
           } else {
             RSK[start + m +z] = 1 << (m*2);
           }
       }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    
   // PROCESS: getSuperks
   // Canonical m-mers to minimizers

    nt = ((nm-1)/ts) + 1; // Variables counter, mp, nmp and minimizer are set

   /* Initialization */
   if (x==0) {
     counter = 0;
     mp = 0;
     nmp = 0xFFFFFFFF;
     minimizer = 0xFFFFFFFF;
   }

  barrier(CLK_LOCAL_MEM_FENCE);

  start_zone = 0;
  start_superk = 0;

  // Choosing the tile size for the reduction algorithm used in this process
  if (k <= 36) {
    nt3 = 6;
  } else if (k <= 64) {
    nt3 = 8;
  } else if (k <= 100) {
    nt3 = 10;
  } else {
    nt3 = 12;
  }

  // Evaluating the first k-mers of each read

  /* To find the minimizer of the first k-mers of each reading a two-stage atomic reduction methodology is used:
   * - First, the canonical m-mers of the k-mer are grouped into a number of sets equal to nt3; for each set an
   *  atomic operation is performed to find the minimum canonical m-mer of each set.
   * - Secondly, the minimums found in the previous step are reduced to one minimum, which is the minimizer of the k-mer. */


  if (x<nmk) {
    p = x;
    b = RSK[p+m-1]; // Each thread copies its canonical m-mer to b
  }

  if (x < nt3){
    MT[x] = b; // The first “nt3” canonical m-mers are copied to MT
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (x >= nt3 && x<nmk) {
    atomic_min(&MT[x%nt3], b);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (x<nt3) {
    atomic_min(&minimizer, MT[x]);
  }


  barrier(CLK_LOCAL_MEM_FENCE);

  if ((x<nmk) && (b==minimizer)){
    atomic_max(&mp, p);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Processing the remaining k-mers
     while ((start_zone+nmk-1)<(nm-1)) {
     
    flag = false; // Flag to indicate if the position of the current minimizer is equal to start_zone (Start of the last k-mer evaluated)
    // If the position of the current minimizer is equal to start_zone, the flag is set to true and start_zone is incremented by one
   
   if ((x<nmk) && (mp == start_zone)){
      flag = true;
      start_zone = start_zone + 1;
      if (p < start_zone) {
        p = p + nmk;
        b = RSK[p + m - 1];
      }
    }
    /* If the flag is true it means that the current minimizer is no longer part of the next k-mer to evaluate, as consequence
     * there ends a super k-mer. The code below is to store the representation of the super k-mer in the vector RSK,
     * additionally counter is incremented and star_superk and minimizer are set. */
    if ( (x==0) && flag ) {
      a = (start_zone + k - 1 - start_superk) & 0x000000FF;
      RSK[counter] = ((minimizer<<18) & 0xFFFC0000) | ((start_superk<<8) & 0x0003FF00) | a;
      counter++;
      start_superk  = start_zone;
      minimizer = 0xFFFFFFFF;
    }
    /* If the flag is true, the next k-mer to evaluate does not have minimizer of reference, therefore this k-mer
     * is evaluated in the same way as the first k-mer.*/
    if ((x<nt3) && (flag)) {
      MT[x] = b;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((x >= nt3) && (x<nmk) && flag) {
      atomic_min(&MT[x%nt3], b);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((x < nt3) && flag) {
      atomic_min(&minimizer, MT[x]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((x<nmk) && flag && (b == minimizer)) {
      atomic_max(&mp, p);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    /* If the flag is false it means that the current minimizer is in a different position to the start of the last k-mer evaluated,
     * therefore it is possible that the following k-mers share the same minimizer
     *
     * From all the next k-mers that have the possibility to share the same minimizer only the last one is evaluated: its canonical
     * m-mers (different to the first) are compared with the current minimizer (its first canonical m-mer), if there are canonical
     * m-mers lower than the current minimizer, the new minimizer will be the one with the lowest position. */
    if ((x<nmk) && flag==false) {
      start_zone = mp;
      if (p<start_zone) {
        p = p + nmk;
        if ( p < (nm -1)) {
          b = RSK[p+m-1];
          if (b < minimizer) {
            atomic_min(&nmp, p);
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    /* If a new minimizer is found, it means that a super k-mer ends. The code below is to stor the representation of the
     * super k-mer in the vector RSK, additionally counter is incremented and star_superk, minimizer, mp and nmp are set. */
    if ((x==0) && (flag==false) && (nmp != 0xFFFFFFFF)) {
      a = (nmp - start_superk + m - 1) & 0x000000FF;
      RSK[counter] = ((minimizer <<  18) & 0xFFFC0000) | ((start_superk << 8) & 0x0003FF00) | a;
      mp = nmp;
      minimizer = RSK[mp+m-1];
      counter++;
      start_superk = mp - nmk + 1;
      nmp = 0xFFFFFFFF;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  /* When each read is fully evaluated, it is necessary to store the representation of last super k-mer in the vector RSK */

  if (x==0) {
    a = (nm - start_superk + m - 1) & 0x000000FF;
    RSK[counter] = ((minimizer << 18) & 0xFFFC0000) | ((start_superk<<8) & 0x0003FF00) | a;
    counter ++;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // PROCESS: local2Global
  // Results in local memory to Results in global memory
  ts = get_local_size(0);  // ts: Tile size, work group size
  nt4 = (r-1)/ts + 1; // nt: Number of tiles

  if ((x<ts) && (x<counter) && (y<nr)) {
    for (int i=0; i<nt4; i++){  // coalesced
      p=ts*i+x;
      if (p > counter-1)
      {
        break;
      }
      RSK_G[y*r+p] = RSK[p];
    }
  }

  // Extract counters to global mem
  if ((x == 0) && (y<nr))
  {
    counters[y] = counter;
  }
}
