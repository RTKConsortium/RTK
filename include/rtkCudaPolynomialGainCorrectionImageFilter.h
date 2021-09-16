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

#ifndef rtkCudaPolynomialGainCorrectionImageFilter_h
#define rtkCudaPolynomialGainCorrectionImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkPolynomialGainCorrectionImageFilter.h"
#  include "RTKExport.h"

#  include <itkCudaImage.h>
#  include <itkCudaImageToImageFilter.h>

#  include "rtkConfiguration.h"

namespace rtk
{

/** \class PolynomialGainCorrectionImageFilter
 * \brief Cuda version of PolynomialGainCorrectionImageFilter.
 *
 * Cuda version of PolynomialGainCorrectionImageFilter.
 *
 * \see PolynomialGainCorrectionImageFilter
 *
 * \author Sebastien Brousmiche
 *
 * \ingroup RTK
 */
class RTK_EXPORT CudaPolynomialGainCorrectionImageFilter
  : public itk::CudaImageToImageFilter<
      itk::CudaImage<unsigned short, 3>,
      itk::CudaImage<float, 3>,
      PolynomialGainCorrectionImageFilter<itk::CudaImage<unsigned short, 3>, itk::CudaImage<float, 3>>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaPolynomialGainCorrectionImageFilter);

  /** Convenience type alias **/
  using ImageType = itk::CudaImage<float, 3>;
  using CPUPolyGainFilterType =
    PolynomialGainCorrectionImageFilter<itk::CudaImage<unsigned short, 3>, itk::CudaImage<float, 3>>;

  /** Standard class type alias. */
  using Self = CudaPolynomialGainCorrectionImageFilter;
  using Superclass = itk::CudaImageToImageFilter<ImageType, ImageType, CPUPolyGainFilterType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaPolynomialGainCorrectionImageFilter, itk::CudaImageToImageFilter);

protected:
  /** Standard constructor **/
  CudaPolynomialGainCorrectionImageFilter();
  /** Destructor **/
  virtual ~CudaPolynomialGainCorrectionImageFilter();

  virtual void
  GPUGenerateData();
};

} // namespace rtk

#endif // end conditional definition of the class

#endif // rtkCudaPolynomialGainCorrectionImageFilter_h
