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
#ifndef rtkCudaZengBackProjectionImageFilter_h
#define rtkCudaZengBackProjectionImageFilter_h

#include "rtkConfiguration.h"
#ifdef RTK_USE_CUDA

#  include "rtkZengBackProjectionImageFilter.h"
#  include "RTKExport.h"
#  include <itkCudaImage.h>
#  include <itkCudaInPlaceImageFilter.h>
#  include <itkImageBase.h>

namespace rtk
{

/** \class CudaZengBackProjectionImageFilter
 * \brief CUDA implementation of the rotation-based Zeng backprojector.
 *
 * The implementation reproduces the slice recursion of
 * ZengBackProjectionImageFilter, including the depth-dependent Gaussian PSF
 * and the optional attenuation map (input 2).
 * See Zeng et al., IEEE Transactions on Medical Imaging, 1999,
 * doi:10.1109/42.796285.
 *
 * \ingroup RTK Projector CudaImageToImageFilter
 */
class RTK_EXPORT CudaZengBackProjectionImageFilter
  : public itk::CudaInPlaceImageFilter<
      itk::CudaImage<float, 3>,
      itk::CudaImage<float, 3>,
      ZengBackProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaZengBackProjectionImageFilter);
  using ImageType = itk::CudaImage<float, 3>;
  /** Metadata-only image used to build the rotated-grid transforms. Using
   * ImageBase avoids the CUDA image factory replacing itk::Image::New(). */
  using CPUImageType = itk::ImageBase<3>;
  using ProjectorType = ZengBackProjectionImageFilter<ImageType, ImageType>;
  using Self = CudaZengBackProjectionImageFilter;
  using Superclass = itk::CudaInPlaceImageFilter<ImageType, ImageType, ProjectorType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaZengBackProjectionImageFilter);

protected:
  CudaZengBackProjectionImageFilter();
  ~CudaZengBackProjectionImageFilter() override;
  void
  GPUGenerateData() override;

private:
  void * m_CudaWorkspace{ nullptr };
};

} // namespace rtk
#endif // RTK_USE_CUDA
#endif // rtkCudaZengBackProjectionImageFilter_h
