#include "itkCudaUtilities.hcu"
#include <itkMacro.h>

void
CUDA_check_error (const std::string &msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
      itkGenericExceptionMacro(<< "CUDA ERROR: " << msg << " (" << err << ")." << std::endl);
}
