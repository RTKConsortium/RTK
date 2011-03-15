#include "itkFFTWCommon.h"

itk::FastMutexLock::Pointer itk::fftw::Proxy<float>::lock = NULL;
itk::FastMutexLock::Pointer itk::fftw::Proxy<double>::lock = NULL;
   
