__kernel void getSuperK_M(
   __global uint* RSK_G, // Matrix of reads
   __global uint* counters, // Matrix of temporal output
   const unsigned int nr, // Number of reads
   const unsigned int r, // Read size
   const unsigned int k, // Kmer size
   const unsigned int m // m-mer size
   )
{
   uint x, y, yl, xl, p, start, offset, nmk, nm, min, a, b, c, d, idt, ts, nt, lsd, nkr, tmp, ls, start_tile, start_superk, nt1, nt2, nt3, nt4 = 0;
   bool flag;

   x = get_global_id(0);
   y = get_global_id(1);
   xl = get_local_id(0);
   yl = get_local_id(1);

  __local uint RSK[180]; // Vector to store reads, m-mers and superk-mers. This vector is overwritten in each stage, size: length of the read

  __local uint RCMT[7]; // Reverse complement of current m-mer of tile (32 bits), size: nt2
  __local uint MT[7]; // Current m-mer of tile (32 bits), size: Greater between nt2 and nt3
  __local uint counter, mp, nmp, minimizer; // minimizer position, new minimizer  position, minimizer

   nmk = k - m + 1;
   // Global to local
   // Reads in global memory to Read in local memory
   
   ts = get_local_size(0);
   nt1 = (r-1)/ts + 1; // nt: Number of tiles or number of sub-reads per read
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

    // FUNCTION: GetM-mers
    // Read to decimal representation of m-mers
    lsd = get_local_size(0) / m; // lsd: Local space divisions
    nm = r - m + 1; // nm: Number of m-mers per read
    ts = ((nm-1)/lsd) + 1; // ts: Tile size for this process
    nt2 = ((nm-1)/ts) + 1 ; // nt2: Number of tiles for this process
    // a = (int) (pow(((double) 2, (double) (2*m)) - 1); // a:Mask (Sample: 63 for m = 4, to cancel all bits different to first 6)
    a = 4095; // Mask   2**(2m-2)cc

    if ((x < m*nt2) && (y<nr)) {
      // Parallel computing of the first m-mer of each tile
       
      idt = x/m; // Tile id
      start = ts*idt; // Position of the first base for each tile
      offset = x % m;
      p = (start) + (offset); // Position of the base corresponding a the thread
      MT[idt] = 0; MT Reset
      RCMT[idt] = 0; // RCMT Reset
      b = RSK[p]; // Each thread copies its base to b
      atomic_or(&MT[idt], (b << (m-(x%m)-1)*2)); // Calculating the first m-mer of each tile  
      atomic_or(&RCMT[idt], (((~b) & 3) << (x%m)*2)); // Calculating the reverse complement of the first m-mer of each tile
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((x<nt2) && (y<nr)) { // Finding the first canonical m-mer of each tile
      // The canonical m-mers of a read are saved in the same vector where the read was stored (RSK)
      /* Each canonical m-mer is saved in the position where was its last base (This avoid the over-write of bases that Will be used by another tile) */
      idt = x;
      start = ts*idt;
      if (MT[idt]<RCMT[idt]){
        RSK[start + m -1] = MT[idt];
      } else {
      	RSK[start + m -1] = RCMT[idt];
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
           c = ((c & a) << 2)|b;  // Calculating the current m-mer from the previous m-mer and the new base read (b)
           d = ((d>>2) & (a)) | (((~b) & 3)<<((m-1)*2)); // Calculating the reverse complement of the current m-mer from the reverse complement of the previous m-mer and the new base read (b)
           if (c<d)	{  // Finding the canonical m-mer
             RSK[start+m+z]=c;
            } else {
             RSK[start+m+z]=d;
            }
           }
      }

      barrier(CLK_LOCAL_MEM_FENCE);
   
   // PROCESS: getSuperks
   // Canonical m-mers to minimizers


      nt = ((nm-1)/ts) + 1; // nt: Number of tiles or number of sub-reads per read


   /* Initialization */
   if (x==0) {
     counter = 0;
     mp = 0;
     nmp = 0xFFFFFFFF;
     minimizer = 0xFFFFFFFF;
   }

  barrier(CLK_LOCAL_MEM_FENCE);

  start_tile = 0; // Start of the currently evaluated zone
  start_superk = 0;
  if (k <= 36) {  // Choosing the tile size for the reduction algorithm used in this process
    nt3 = 6;
  } else if (k <= 64) {
    nt3 = 8;
  } else if (k <= 100) {
    nt3 = 10;
  } else {
    nt3 = 12;
  }
   
   // Evaluating the first k-mers of each read

/* To find the minimizer of the first k-mers of each reading a two-stage atomic reduction methodology is used: First, the canonical m-mers of the k-mer are grouped into a number of sets equal to nt3; for each set an atomic operation is performed to find the minimum canonical m-mer of each set. Secondly, the minimums found in the previous step are reduced to one minimum, which is the minimizer of the k-mer. */  

   

  if (x<nmk) {
    p = x;
    b = RSK[p+m-1];  // Each thread copies its canonical m-mer to b
  }

  if (x < nt3){
    MT[x] = b;  // The first “nt3” canonical m-mers are copied to MT
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
   
  while ((start_tile+nmk-1)<(nm-1)) {
    flag = false;  // Flag to indicate if the position of the current minimizer is equal to start_zone (Start of the last k-mer evaluated)
    if ((x<nmk) && (mp == start_tile)){
      // If the position of the current minimizer is equal to start_zone, the flag is set to true and start_zone is incremented by one
      flag = true;
      start_tile = start_tile + 1;
      if (p < start_tile) {
        p = p + nmk;
        b = RSK[p + m - 1];
      }
    }

    if ( (x==0) && flag ) {
       
       /* If the flag is true it means that the current minimizer is no longer part of the next k-mer to evaluate, as consequence there ends a super k-mer. The code below is to store the representation of the super k-mer in the vector RSK, additionally counter is incremented and star_superk and minimizer are set. */ 
       
      a = (start_tile + k - 1 - start_superk) & 0x000000FF;
      RSK[counter] = ((minimizer<<18) & 0xFFFC0000) | ((start_superk<<8) & 0x0003FF00) | a;
      counter++;
      start_superk  = start_tile;
      minimizer = 0xFFFFFFFF;
    }

     /* If the flag is true, the next k-mer to evaluate does not have minimizer of reference, therefore this k-mer is evaluated in the same way as the first k-mer.*/
     
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

    /* If the flag is false it means that The current minimizer is in a different position to the start of the last k-mer evaluated, therefore it is possible that the following k-mers share the same minimizer */  

   /* From all the next k-mers that have the possibility to share the same minimizer only the last one is evaluated: its canónical m-mers (different to the first) are compared with the current minimizer (its first canonical m-mer), if there are canonical m-mers lower than the current minimizer, the new minimizer will be the one with the lowest position. */

     
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((x<nmk) && flag==false) {
      start_tile = mp;
      if (p<start_tile) {
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
     
     /* If a new minimizer is found, it means that a super k-mer ends. The code below is to save the representation of the super k-mer in the vector RSK, additionally counter is incremented and star_superk, minimizer, mp and nmp are set. */


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
   
   /* When each read is fully evaluated, it is necessary to store the representation of last super k-mer in the vector RSK  */ 



  if (x==0) {
    a = (nm - start_superk + m - 1) & 0x000000FF;
    RSK[counter] = ((minimizer << 18) & 0xFFFC0000) | ((start_superk<<8) & 0x0003FF00) | a;
    counter ++;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

   // PROCESS: local2Global
    //Reads in local memory to Read in global memory
   
    ts = get_local_size(0);  // ts: Tile size, work group size
    nt4 = (r-1)/ts + 1; // nt: Number of tiles for this process 

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

   // Extract counters
    if ((x == 0) && (y<nr))
     {
             counters[y] = counter;
      }
}
