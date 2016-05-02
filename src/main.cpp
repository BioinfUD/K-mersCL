#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#define TIPO_DISPOSITIVO CL_DEVICE_TYPE_DEFAULT



int main ( int argc, char *argv[] ){

  if ( argc != 4 ) // argc should be 2 for correct execution
  // We print argv[0] assuming it is the program name
  cout<< "Kmers-CL: Extraer kmers de lecturas "
  cout<<"Uso: "<< argv[0] << "<Filename>" << "<<read_size>" <<  "<kmer_size>" << "\n";
else {
  std::string file_name(argv[1]);
  int rs = atoi(argv[2]);
  int ks = atoi(argv[3]);
  int numero_lecturas = 1000;

  //  Para almacenar lecturas
  cl::Buffer d_r;
  // Para almacenar kmers
  cl::Buffer d_k;
  // Cada char es un nucleotido en las lecturas
  char reads[] = "TATCGACTAGCTACGTACGTAGCTAGCTAGCGTACGATCGTACGGTACGTAGCATCGATCGATCGAGCTGACTAGGCTGACTA";

  cl::Context contexto(TIPO_DISPOSITIVO);
  cl::Program programa(contexto, utilidades::cargarPrograma("getKmers.cl"), true);
  cl::CommandQueue cola(contexto);
  d_r  = cl::Buffer(contexto, begin(reads), end(reads), true);
  d_k  = cl::Buffer(contexto, CL_MEM_WRITE_ONLY, sizeof(char) * numero_lecturas*k*(rs-ks+1));
  auto getkmers = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(programa, "getkmers");
  // Numero de kmers a extraer
  int contador = numero_lecturas*(rs-ks+1);
  cl::NDRange rangondimensional(contador);
  cl::EnqueueArgs argumentos(
      cola, rangondimensional);
  getkmers(  argumentos,
          d_r, d_k, ks, rs, numero_lecturas, contador
      );
  cola.finish();





}
