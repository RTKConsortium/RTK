#include "itkFFTWCommon.h"

#if defined(USE_FFTWF)
itk::FastMutexLock::Pointer itk::fftw::Proxy<float>::lock = NULL;
#endif

#if defined(USE_FFTWD)
itk::FastMutexLock::Pointer itk::fftw::Proxy<double>::lock = NULL;
#endif
   
