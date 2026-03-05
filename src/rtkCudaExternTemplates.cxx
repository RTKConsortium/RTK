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

#include "rtkCudaExternTemplates.h"

#ifdef RTK_USE_CUDA

// ITK base class explicit instantiation definitions
template class itk::ImageSource<itk::CudaImage<float, 3>>;
template class itk::ImageToImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>;
template class itk::InPlaceImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>;
template class itk::ImageSource<itk::CudaImage<itk::CovariantVector<float, 3>, 3>>;

// RTK base class explicit instantiation definitions
#  include "rtkBackProjectionImageFilter.h"
#  include "rtkFDKBackProjectionImageFilter.h"
#  include "rtkForwardProjectionImageFilter.h"
#  include "rtkDisplacedDetectorImageFilter.h"
#  include "rtkFDKWeightProjectionFilter.h"
#  include "rtkConstantImageSource.h"
#  include "rtkInterpolatorWithKnownWeightsImageFilter.h"

template class rtk::BackProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>;
template class rtk::FDKBackProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>;
template class rtk::ForwardProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>;
template class rtk::DisplacedDetectorImageFilter<itk::CudaImage<float, 3>>;
template class rtk::FDKWeightProjectionFilter<itk::CudaImage<float, 3>>;
template class rtk::ConstantImageSource<itk::CudaImage<float, 3>>;
template class rtk::InterpolatorWithKnownWeightsImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 4>>;

namespace rtk
{
// Empty namespace block required by KWStyle
} // namespace rtk

#endif // RTK_USE_CUDA
