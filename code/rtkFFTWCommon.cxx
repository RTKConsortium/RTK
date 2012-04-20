#include "rtkFFTWCommon.h"

#if defined(USE_FFTWF)
itk::FastMutexLock::Pointer rtk::fftw::Proxy<float>::lock = NULL;
#endif

#if defined(USE_FFTWD)
itk::FastMutexLock::Pointer rtk::fftw::Proxy<double>::lock = NULL;
#endif
