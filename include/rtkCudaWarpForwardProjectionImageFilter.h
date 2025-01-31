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

#ifndef rtkCudaWarpForwardProjectionImageFilter_h
#define rtkCudaWarpForwardProjectionImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkForwardProjectionImageFilter.h"
#  include "itkCudaInPlaceImageFilter.h"
#  include "itkCudaUtil.h"
#  include "RTKExport.h"

/** \class CudaWarpForwardProjectionImageFilter
 * \brief Trilinear interpolation forward projection in warped volume implemented in CUDA
 *
 * CudaWarpForwardProjectionImageFilter is similar to
 * CudaForwardProjectionImageFilter, but assumes the object has
 * undergone a known deformation, and compensates for it during the
 * forward projection. It amounts to bending the trajectories of the rays.
 *
 * \author Cyril Mory
 *
 * \ingroup RTK Projector CudaImageToImageFilter
 */

namespace rtk
{

class RTK_EXPORT CudaWarpForwardProjectionImageFilter
  : public itk::CudaInPlaceImageFilter<itk::CudaImage<float, 3>,
                                       itk::CudaImage<float, 3>,
                                       ForwardProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaWarpForwardProjectionImageFilter);

  /** Standard class type alias. */
  using InputImageType = itk::CudaImage<float, 3>;
  using DisplacementVectorType = itk::CovariantVector<float, 3>;
  using Self = CudaWarpForwardProjectionImageFilter;
  using Superclass = ForwardProjectionImageFilter<InputImageType, InputImageType>;
  using GPUSuperclass = itk::CudaInPlaceImageFilter<InputImageType, InputImageType, Superclass>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using DVFType = itk::CudaImage<DisplacementVectorType, 3>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(CudaWarpForwardProjectionImageFilter);

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
  SetDisplacementField(const DVFType * DVF);
  DVFType::Pointer
  GetDisplacementField();

  /** Set step size along ray (in mm). Default is 1 mm. */
  itkGetConstMacro(StepSize, double);
  itkSetMacro(StepSize, double);

protected:
  CudaWarpForwardProjectionImageFilter();
  ~CudaWarpForwardProjectionImageFilter() {};

  virtual void
  GenerateInputRequestedRegion();

  virtual void
  GPUGenerateData();

private:
  double m_StepSize;
}; // end of class

} // end namespace rtk

#endif // end conditional definition of the class

#endif
