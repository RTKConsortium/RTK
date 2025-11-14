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

#ifndef rtkCudaForwardProjectionImageFilter_h
#define rtkCudaForwardProjectionImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkForwardProjectionImageFilter.h"
#  include "itkCudaInPlaceImageFilter.h"
#  include "itkCudaUtil.h"
#  include "RTKExport.h"

/** \class CudaForwardProjectionImageFilter
 * \brief Trilinear interpolation forward projection implemented in CUDA
 *
 * CudaForwardProjectionImageFilter is similar to
 * JosephForwardProjectionImageFilter, except it uses a
 * fixed step between sampling points instead of placing these
 * sampling points only on the main direction slices.
 *
 * The code was developed based on the file tt_project_ray_gpu_kernels.cu of
 * NiftyRec (https://sourceforge.net/projects/niftyrec/) which is distributed under a BSD
 * license. See COPYRIGHT.TXT.
 *
 * \author Marc Vila, updated by Simon Rit and Cyril Mory
 *
 * \ingroup RTK Projector CudaImageToImageFilter
 */

namespace rtk
{

template <class TInputImage = itk::CudaImage<float, 3>, class TOutputImage = TInputImage>
class ITK_TEMPLATE_EXPORT CudaForwardProjectionImageFilter
  : public itk::
      CudaInPlaceImageFilter<TInputImage, TOutputImage, ForwardProjectionImageFilter<TInputImage, TOutputImage>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaForwardProjectionImageFilter);

  /** Standard class type alias. */
  using Self = CudaForwardProjectionImageFilter;
  using Superclass = ForwardProjectionImageFilter<TInputImage, TOutputImage>;
  using GPUSuperclass = itk::CudaInPlaceImageFilter<TInputImage, TOutputImage, Superclass>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using VectorType = itk::Vector<float, 3>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(CudaForwardProjectionImageFilter);

  /** Set step size along ray (in mm). Default is the min value of spacing. */
  itkGetConstMacro(StepSize, double);
  itkSetMacro(StepSize, double);

protected:
  CudaForwardProjectionImageFilter() = default;
  ~CudaForwardProjectionImageFilter() {};

  virtual void
  GPUGenerateData();

private:
  double m_StepSize{ 0. };
}; // end of class

} // end namespace rtk

#  ifndef ITK_MANUAL_INSTANTIATION
#    include "rtkCudaForwardProjectionImageFilter.hxx"
#  endif

#endif // end conditional definition of the class

#endif
