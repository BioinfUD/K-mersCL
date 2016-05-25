__kernel void N2B64(
   __global uchar* SK_4, // 8 bits (1 kmer por posicion) entrada // # DE COLUMNAS, DEBE SER MULTIPLO DE 4. SI K NO ES MULTIPLO DE 4, SE DEBE AGREGAR LAS COLUMNAS INICIALES IZQUIERDA= CON 0
   __global ulong* TMP, // Matriz intermedia donde se guardan corrimientos
   __global ulong* SK_2_64, // 8 bits (4 kmers) salida, n cols = c_sk_4 / 4
   const unsigned int c_sk_4, // Numero de columnas de SK_4
   const unsigned int s)
{

   int i = get_global_id(0);
   int j = get_global_id(1);


   if((i <= c_sk_4-1)&& (j<=s-1))  {

     TMP[(j*c_sk_4)+i] = (ulong)SK_4[(j*c_sk_4)+i] << (64 - 2*((i%32)+1));

    }



    int c_tmp = c_sk_4;
      // 2, log_2(n4), n=1 dado que n8bits, c=ncols
    for(int c=32; c>=2; c/=2) {
      c_tmp /= 2;
      if((i<=(c-1)) && (j<=s-1)) {
        TMP[(j*c_sk_4)+i] =  TMP[(j*c_sk_4)+i*2] | TMP[(j*c_sk_4)+i*2+1];
      }
    } 

    if((i <=((c_sk_4/32))-1)&& (j<=s-1))  {
      SK_2_64[(j*c_sk_4/32)+i] = TMP[(j*c_sk_4)+i];
    }
  }
