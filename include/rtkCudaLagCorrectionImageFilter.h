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

#ifndef rtkCudaLagCorrectionImageFilter_h
#define rtkCudaLagCorrectionImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkLagCorrectionImageFilter.h"
#  include "RTKExport.h"

#  include <itkCudaImage.h>
#  include <itkCudaInPlaceImageFilter.h>

#  include "rtkConfiguration.h"

namespace rtk
{

/** \class CudaLagCorrectionImageFilter
 * \brief Cuda version of LagCorrectionImageFilter.
 *
 * Cuda version of LagCorrectionImageFilter.
 *
 * \see LagCorrectionImageFilter
 *
 * \author Sebastien Brousmiche
 *
 * \ingroup RTK
 */
class RTK_EXPORT CudaLagCorrectionImageFilter
  : public itk::CudaInPlaceImageFilter<itk::CudaImage<unsigned short, 3>,
                                       itk::CudaImage<unsigned short, 3>,
                                       LagCorrectionImageFilter<itk::CudaImage<unsigned short, 3>, 4>>
{
public:
#  if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(CudaLagCorrectionImageFilter);
#  else
  ITK_DISALLOW_COPY_AND_MOVE(CudaLagCorrectionImageFilter);
#  endif

  /** Convenience type alias **/
  using ImageType = itk::CudaImage<unsigned short, 3>;
  using CPULagFilterType = LagCorrectionImageFilter<ImageType, 4>;

  /** Standard class type alias. */
  using Self = CudaLagCorrectionImageFilter;
  using Superclass = itk::CudaInPlaceImageFilter<ImageType, ImageType, CPULagFilterType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaLagCorrectionImageFilter, itk::CudaInPlaceImageFilter);

protected:
  /** Standard constructor **/
  CudaLagCorrectionImageFilter();
  /** Destructor **/
  virtual ~CudaLagCorrectionImageFilter();

  virtual void
  GPUGenerateData();
};

} // namespace rtk

#endif // end conditional definition of the class

#endif // rtkCudaLagCorrectionImageFilter_h
