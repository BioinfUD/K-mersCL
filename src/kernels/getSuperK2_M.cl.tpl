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

  __local uint RSK[READ_SIZE]; // Vector of a read and super k-mers (32 bits) , len = lenght of reads

  __local uint RCMT[SECOND_MT]; // Position of minimizer in each tile, (nm-1)/(ts)   +  1, ts -> (nm-1)/lsd   + 1, nm=r-m-1, lsd=localSpaceSize/m
  __local uint MT[NT_MAX]; // max(nt1, nt2) , max(nt anterior , nt nuevo)
  __local uint counter, mp, nmp, minimizer; // minimizer position, new minimizer  position, minimizer
   // Mask to use when compacting
    a = A_MASK; // Mask

   nmk = k - m + 1;
   // Global to local
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
    lsd = get_local_size(0) / m;
    nm = r - m + 1; // nm: Number of m-mers per read
    ts = ((nm-1)/lsd) + 1; // ts: Tile size
    nt2 = ((nm-1)/ts) + 1 ; // nt: Number of tiles or number of sub-reads per read

    if ((x < m*nt2) && (y<nr)) {
      // Cómputo en pllo del primer m-mer de cada tile
      idt = x/m;
      start = ts*idt;
      offset = x % m;
      p = (start) + (offset);
      MT[idt] = 0;
      RCMT[idt] = 0;
      b = RSK[p];
      atomic_or(&MT[idt], (b << (m-(x%m)-1)*2));
      atomic_or(&RCMT[idt], (((~b) & 3) << (x%m)*2));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((x<nt2) && (y<nr)) {
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
