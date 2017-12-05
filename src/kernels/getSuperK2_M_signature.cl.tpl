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

    // Checking for signature
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

    // PROCESS: local2Global
    // Results in local memory to Results in global memory
    ts = get_local_size(0);  // ts: Tile size, work group size
    nt4 = (r-1)/ts + 1; // nt: Number of tiles

    if ((x==0) && (y<nr)) {
      for (int w = 0; w<r; w++) {
        RSK_G[y*r + w] = RSK[w];
      }
    }

}
