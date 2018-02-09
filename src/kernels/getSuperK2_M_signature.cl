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

  __local uint RSK[180]; //  Vector to store reads, m-mers and superk-mers. This vector is overwritten in each stage, size: length of the read

  __local uint RCMT[7]; // Reverse complement of current m-mer of tile (32 bits), size: nt2
  __local uint MT[7]; // Current m-mer of tile (32 bits), size: Greater between nt2 and nt3
  __local uint counter, mp, nmp, minimizer; // minimizer position, new minimizer  position, minimizer

   nmk = k - m + 1;
  
  // PROCESS: global2Local
  // Reads in global memory to Read in local memory

   
   ts = get_local_size(0); ts: Tile size, work group size
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
   
   
    lsd = get_local_size(0) / m; lsd: Local space divisions
    nm = r - m + 1; // nm: Number of m-mers per read
    ts = ((nm-1)/lsd) + 1; // ts: Tile size for this process
    nt2 = ((nm-1)/ts) + 1 ; // nt2: Number of tiles for this process
    // a = (int) (pow(((double) 2, (double) (2*m)) - 1); // a:Mask (Sample: 63 for m = 4, to cancel all bits different to first 6) 
    a = 4095; // Mask   2**(2m-2)cc

   // Parallel computing of the first m-mer of each tile
   
    if ((x < m*nt2) && (y<nr)) {
      // Cómputo en pllo del primer m-mer de cada tile
      idt = x/m; // Tile id
      start = ts*idt; // Position of the first base for each tile
      offset = x % m;
      p = (start) + (offset); // Position of the base corresponding a the thread
      MT[idt] = 0; // RCMT Reset
      RCMT[idt] = 0; // Each thread copies its base to b
      b = RSK[p];
      atomic_or(&MT[idt], (b << (m-(x%m)-1)*2)); // Calculating the first m-mer of each tile 
      atomic_or(&RCMT[idt], (((~b) & 3) << (x%m)*2)); // Calculating the reverse complement of the first m-mer of each tile
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((x<nt2) && (y<nr)) {
      // Finding the first canonical m-mer of each tile
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

      // Cómputo en serie del resto de m-mers de cada tile
    if ((x<nt2) && (y<nr))	{
       idt = x;
       start = ts*idt;
       c = MT[idt]; // Almacena en C el valor decimal del primer m-mer del tile
       d = RCMT[idt];

       for (int z=0; z<(ts-1); z++) {
       		if (start+m+z > r-1) {
            break;
          }
           b = RSK[start+m+z]; // Obtiene la siguiente base
           c = ((c & a) << 2)|b; // Obtiene el valor decimal del m-mer (A partir del resultado   // anterior y la nueva base)
           d = ((d>>2) & (a)) | (((~b) & 3)<<((m-1)*2));
           if (c<d)	{
             RSK[start+m+z]=c;
            } else {
             RSK[start+m+z]=d;
            }
           }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      nt = ((nm-1)/ts) + 1; // nt: Number of tiles or number of sub-reads per read


   /* Initialization */
   if (x==0) {
     counter = 0;
     mp = 0;
     nmp = 0xFFFFFFFF;
     minimizer = 0xFFFFFFFF;
   }

  barrier(CLK_LOCAL_MEM_FENCE);

  start_tile = 0;
  start_superk = 0;
  if (k <= 36) {
    nt3 = 6;
  } else if (k <= 64) {
    nt3 = 8;
  } else if (k <= 100) {
    nt3 = 10;
  } else {
    nt3 = 12;
  }

  if (x<nmk) {
    p = x;
    b = RSK[p+m-1];
  }

  if (x < nt3){
    MT[x] = b;
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

  // The reamaining k-mers
  while ((start_tile+nmk-1)<(nm-1)) {
    flag = false;
    if ((x<nmk) && (mp == start_tile)){
      flag = true;
      start_tile = start_tile + 1;
      if (p < start_tile) {
        p = p + nmk;
        b = RSK[p + m - 1];
      }
    }

    if ( (x==0) && flag ) {
      a = (start_tile + k - 1 - start_superk) & 0x000000FF;
      RSK[counter] = ((minimizer<<18) & 0xFFFC0000) | ((start_superk<<8) & 0x0003FF00) | a;
      counter++;
      start_superk  = start_tile;
      minimizer = 0xFFFFFFFF;
    }

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

  if (x==0) {
    a = (nm - start_superk + m - 1) & 0x000000FF;
    RSK[counter] = ((minimizer << 18) & 0xFFFC0000) | ((start_superk<<8) & 0x0003FF00) | a;
    counter ++;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

    //FUNCTION: Local2Global
    //Reads in local memory to Read in global memory
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

   // Extract counters
    if ((x == 0) && (y<nr))
     {
             counters[y] = counter;
      }
}
