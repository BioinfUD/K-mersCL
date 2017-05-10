__kernel void getSuperK_M(
   __global uint* R2M_G, // Matrix of reads
   __global uint* counters, // Matrix of temporal output
   __global uint* TMP, // Matrix of temporal output
   const unsigned int nr, // Number of reads
   const unsigned int r, // Read size
   const unsigned int k, // Kmer size
   const unsigned int m // m-mer size
   )
{
   uint x, y, yl, xl, p, start, offset, nmk, nm, min, a, b, c, idt, ts, nt, lsd, nkr, tmp, ls, d = 0;
   bool flag;

   x = get_global_id(0);
   y = get_global_id(1);
   xl = get_local_id(0);
   yl = get_local_id(1);
   ls = get_local_size(0);


  __local uint R2M_L[180]; // Vector of a read and super k-mers (32 bits) , len = lenght of reads
  __local uint mR_10[180-4+1]; // Vector containing mmers, len = lenght of reads - minimizer size + 1
  __local uint counter; // Size of mmers
  __local uint MT[9]; // Position of minimizer in each tile
  __local uint PMT[9]; // Current minimizer in each tile

  if (x==0) {
    counter = 0;
  }

   // Global to local
   ts = ls;
   nt = (r-1)/ts + 1; // nt: Number of tiles or number of sub-reads per read
   if ((x<ts) && (x<r) && (y<nr)){
      for (int i=0; i<nt; i++) { // coalesced
        p=ts*i+x;
        if (p > r-1) {
          break;
        }
        R2M_L[p]=R2M_G[y*r+p];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // FUNCTION: GetM-mers
    // Read to decimal representation of m-mers
    lsd = ls / m;
    nm = r - m + 1; // nm: Number of m-mers per read
    ts = ((nm-1)/lsd) + 1; // ts: Tile size
    nt = ((nm-1)/ts) + 1 ; // nt: Number of tiles or number of sub-reads per read
    a = 63; // a:Mask , same as: 2^(2m-2)-1  m=4

    // rewrited from here
    if ((x < m*nt) && (y<nr)) {
      // Cómputo en pllo del primer m-mer de cada tile
      idt = x/m;
      start = ts*idt;
      offset = x % m;
      p = (start) + (offset);
      mR_10[start] = 0;
      b = R2M_L [p];
      atomic_or(&mR_10[start], (b<<(m-(x%m)-1)*2));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

      // Cómputo en serie del resto de m-mers de cada tile

      if ((x%m == 0) && (x < m*nt) && (y<nr)) {
        c = mR_10[start];
        for (int z=0; z<(ts-1); z++) {
          if (start+m+z > r-1) {
            break;
          }
          c = (c & a) << 2;
          b = R2M_L[start+m+z];
          c = c | b;
          mR_10[start+1+z]=c;
        }
      }
    barrier(CLK_LOCAL_MEM_FENCE);

   // Clear R2M_L
   ts = ls;  // ts: Tile size, work group size
   nt = (r-1)/ts + 1; // nt: Number of tiles
   if ((x<ts) && (x<r) && (y<nr)){
      for (int i=0; i<nt; i++)  // coalesced
        {
          p=ts*i+x;
          if (p > r-1)
            { break; }
          R2M_L[p] = 0;
        }
    }

   nt = ((nm-1)/ts) + 1; // nt: Number of tiles or number of sub-reads per read

   // Extract MR_10 debuggin
   if ((x<ts) && (y<nr)){
     for (int i=0; i<nt; i++) { // coalesced
       p=ts*i+x;
       if (p > nm-1) {
         break;
       }
       TMP[y*nm+p]=mR_10[p];
     }
   }

    // FUNCTION: GetSK_M
    // m-mers to minimizers
    nmk = k-m+1; // nm: Number of m-mers per k-mers
    nkr = r - k + 1;
    lsd = ls/nmk;
    ts = (nkr-1)/lsd + 1; // ts: Tile size
    nt = (nkr-1)/ts + 1; // nt: Number of tiles

    if (x < nt){
      MT[x] = 0xFFFFFFFF;
      PMT[x] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

  if ((x<(nt*nmk )) && (y<nr)){
  //if ((x>=(5*nmk)) && (y<nr) && (x<(6*nmk))){
      idt = x/nmk;
      start = ts*idt;
      offset = x%nmk;
      p = start + offset;
      // Cómputo en pllo del min del primer k-mer de cada tile
      b = mR_10[p];
      atomic_min(&MT[idt],b); // Atomic min - se obtiene el min del primer k-mer de cada tile
}

barrier(CLK_LOCAL_MEM_FENCE);

 if ((x < (nt*nmk)) && (y<nr)) {
   if (b == MT[idt]) {
     if (idt == 0){
        PMT[idt] = atomic_max(&PMT[idt],p);
      } else {
        PMT[idt] = atomic_min(&PMT[idt],p);
      };
   }
 }
barrier(CLK_LOCAL_MEM_FENCE);

for (int z=0; z < ts ; z++){ // Cómputo en serie del min del resto de k-mers de cada tile

    if ((start + nmk + z) > (r -1)){
      break;
    }

    if ((x%nmk - z%nmk) == 0)
    {
            b = mR_10[start + nmk + z]; // Se lee el último m-mer del k-mer a evaluar
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((x < (nt*nmk)) && (y<nr))
    {
      flag = false;
      if (PMT[idt] != (start + z))
      {
        flag = true;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if ( (x < (nt*nmk)) && (y<nr) ) {
        if (flag){
          if ((x%nmk - z%nmk) == 0){
            if (b < MT[idt]){
              // Atomic OR - end of super k-mers
              atomic_or(&R2M_L[PMT[idt]], (start + z + k - 1));

              PMT[idt] = start+nmk+z;

              MT[idt] = b;
              // Atomic OR - start of super k-mers
              atomic_or(&R2M_L[PMT[idt]], ((start + z + 1) << 12));
            }
          }
        }
      }
      if ((x < (nt*nmk)) && (y<nr))  {
        if (!flag) {
          if (x%nmk == 0){
            // Atomic OR - end of super k-mer
            atomic_or(&R2M_L[PMT[idt]], (start + z + k - 1));

          }
          MT[idt] = 0xFFFFFFFF;
        }
      }
        barrier(CLK_LOCAL_MEM_FENCE);

        if ((x < (nt*nmk)) && (y<nr))
      	{
          if (!flag)
          {
              atomic_min(&MT[idt], b);
          }
        }
          barrier(CLK_LOCAL_MEM_FENCE);

          if ((x < (nt*nmk)) && (y<nr))
          {
    			     if (!flag)
               {
    	            if (b == MT[idt])
                    {
                      atomic_min(&PMT[idt],p);
                    }
               }
    		  }

        barrier(CLK_LOCAL_MEM_FENCE);
      if ((x < (nt*nmk)) && (y<nr) )
        {
  			if (!flag)
          {
          if (p == PMT[idt]) {
            if (x%nmk <= z%nmk){
              PMT[idt] = start + ((z/nmk)+1)*nmk + (x%nmk);
            } else {
              PMT[idt] = start + (z/nmk)*nmk +(x%nmk);
            }
            // Atomic OR - start of super k-mers
            atomic_or(&R2M_L[PMT[idt]], ((start + z + 1) << 12));
          }

         }
       }
     }
    barrier(CLK_LOCAL_MEM_FENCE);

      if (x == (nt*nmk - 1)){
        // Atomic OR - end of super k-mer
        atomic_or(&R2M_L[PMT[idt]], (r-1));
      }
      barrier(CLK_LOCAL_MEM_FENCE);


    // Compact output
    lsd = ls/m; // lsd: Local space divisions
    nm=r-m+1; // nm: Number of m-mers per read
    ts=(nm-1)/lsd + 1; // ts: Tile size
    nt=(nm-1)/ts + 1; // // nt: Number of tiles
    if ((x<ts) && (x<nm) && (y<nr)) {
   		for (int i=0; i<nt; i++) {
        p=ts*i+x;
        if (p > nm-1)
          { break; }
        b = R2M_L[p];
        if (b != 0)
        {
    	     c = atom_inc(&counter); // Atomic increment
           a = ((b & 0x00000FFF) - ((b >> 12) & 0x00000FFF)) & 0x000000FF; // a is the size
           R2M_L[c] =  ((mR_10[p] << 20) & 0xFFF00000) | ((b >> 4) & 0x000FFF00) | a;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    //   //FUNCTION: Local2Global
    //   //Reads in local memory to Read in global memory

    ts = ls;  // ts: Tile size, work group size
    nt = (r-1)/ts + 1; // nt: Number of tiles

    if ((x<ts) && (x<r) && (y<nr)) {
      for (int i=0; i<nt; i++){  // coalesced
        p=ts*i+x;
        if (p > nm-1)
          {
            R2M_G[y*r+p] = 0;
            break;
          }
        R2M_G[y*r+p] = R2M_L[p];
      }
    }
   // Extract counters
    if ((x == 0) && (y<nr))
     {
             counters[y] = counter;
      }
}
