/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkCudaExternTemplates_h
#define rtkCudaExternTemplates_h

#include "rtkConfiguration.h"
#ifdef RTK_USE_CUDA

#  include <itkCudaImage.h>
#  include <itkImageSource.h>
#  include <itkImageToImageFilter.h>
#  include <itkInPlaceImageFilter.h>
#  include <itkCovariantVector.h>
#  include "RTKExport.h"

// On MSVC, extern template declarations need __declspec(dllimport) to
// reliably suppress implicit instantiation in consumer translation units.
// On the DLL-build side (RTK_EXPORTS defined), extern template declarations
// must be hidden entirely to avoid C4910 warnings caused by dllexport
// propagation from derived CUDA filter classes.
#  if defined(_MSC_VER) && !defined(RTK_EXPORTS)
// Consuming the DLL: extern templates with dllimport suppress LNK2005.
#    define RTK_EXTERN_TEMPLATES
#    define RTK_EXPORT_EXPLICIT RTK_EXPORT
#  endif

// Explicit instantiation definitions are in rtkCudaExternTemplates.cxx.
// RTK base class extern templates are at the end of their own headers.
#  ifdef RTK_EXTERN_TEMPLATES
ITK_GCC_PRAGMA_DIAG_PUSH()
ITK_GCC_PRAGMA_DIAG(ignored "-Wattributes")
extern template class RTK_EXPORT_EXPLICIT itk::ImageSource<itk::CudaImage<float, 3>>;
extern template class RTK_EXPORT_EXPLICIT itk::ImageToImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>;
extern template class RTK_EXPORT_EXPLICIT itk::InPlaceImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>;
extern template class RTK_EXPORT_EXPLICIT itk::ImageSource<itk::CudaImage<itk::CovariantVector<float, 3>, 3>>;
ITK_GCC_PRAGMA_DIAG_POP()
#  endif

namespace rtk
{
// Empty namespace block required by KWStyle
} // namespace rtk

#endif // RTK_USE_CUDA
#endif // rtkCudaExternTemplates_h
