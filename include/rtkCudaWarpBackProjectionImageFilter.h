/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkCudaWarpBackProjectionImageFilter_h
#define rtkCudaWarpBackProjectionImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkBackProjectionImageFilter.h"
#  include "RTKExport.h"

#  include <itkCudaImage.h>
#  include <itkCudaInPlaceImageFilter.h>

namespace rtk
{

/** \class CudaWarpBackProjectionImageFilter
 * \brief Voxel-based backprojection into warped volume implemented in CUDA
 *
 * GPU-based implementation of the voxel-based backprojection, assuming
 * a deformation of the volume.
 *
 * \test rtksarttest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK Projector CudaImageToImageFilter
 */
class RTK_EXPORT CudaWarpBackProjectionImageFilter
  : public itk::CudaInPlaceImageFilter<itk::CudaImage<float, 3>,
                                       itk::CudaImage<float, 3>,
                                       BackProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>>
{
public:
#  if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(CudaWarpBackProjectionImageFilter);
#  else
  ITK_DISALLOW_COPY_AND_MOVE(CudaWarpBackProjectionImageFilter);
#  endif

  /** Standard class type alias. */
  using ImageType = itk::CudaImage<float, 3>;
  using DVFType = itk::CudaImage<itk::CovariantVector<float, 3>, 3>;
  using BackProjectionImageFilterType = BackProjectionImageFilter<ImageType, ImageType>;
  using Self = CudaWarpBackProjectionImageFilter;
  using Superclass = itk::CudaInPlaceImageFilter<ImageType, ImageType, BackProjectionImageFilterType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using OutputImageRegionType = ImageType::RegionType;
  using ProjectionImageType = itk::CudaImage<float, 2>;
  using ProjectionImagePointer = ProjectionImageType::Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaWarpBackProjectionImageFilter, Superclass);

  /** Input projection stack */
  void
  SetInputProjectionStack(const InputImageType * ProjectionStack);
  InputImageType::Pointer
  GetInputProjectionStack();

  /** Input displacement vector field */
  void
  SetInputVolume(const InputImageType * Volume);
  InputImageType::Pointer
  GetInputVolume();

  /** Input displacement vector field */
  void
  SetDisplacementField(const DVFType * MVF);
  DVFType::Pointer
  GetDisplacementField();

protected:
  CudaWarpBackProjectionImageFilter();
  ~CudaWarpBackProjectionImageFilter(){};

  virtual void
  GenerateInputRequestedRegion();

  virtual void
  GPUGenerateData();
};

} // end namespace rtk

#endif // end conditional definition of the class

#endif
