/* Argumentos del kernel
Cada instancia de ejecucion del kernel extrae un kmer
r Vector con caracteres de las lecturas (Concatenadas)
k Vector de salida de los caracteres de los kmer
k_s tamano del kmer
r_s tamano de la lectur
r_n numero de lecturas
count numero de kmers que se extraeran
*/

__kernel void getkmers(
   __global char* r,
   __global char* k,
   const unsigned int k_s,
   const unsigned int r_s,
   const unsigned int r_n,
   const unsigned int count
   )
{
   int i = get_global_id(0);
   // Para que no exista un overflow
    if(i < count)  {
       // Numero de lectura procesado
       int rn = i / r_s;
       // Numero de kmer procesado
       int kn = i / k_s;
       // Escribo los caracteres del kmer en el vector de salida
       int pos_i = (rn+1)*(kn+1)*k;
       int ki;
       for (ki = 0; ki <k; ki ++ ) {
        k[pos_i+ki] = r[rn*kn+ki];
       }

       r[i] = a[i] + b[i] + c[i];
   }
}
