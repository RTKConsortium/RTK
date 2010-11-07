
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "external_dependency.h"

#if defined( _WIN32 ) || defined( _WIN64 )
#  define TEST_LIB_API __declspec(dllimport)
#else
#  define TEST_LIB_API extern
#endif

TEST_LIB_API int doit();

int main( int argc, char **argv )
{
  CHECK_CUDA_ERROR();
  return doit();
};

