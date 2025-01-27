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

#ifndef rtkCudaDisplacedDetectorImageFilter_h
#define rtkCudaDisplacedDetectorImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkDisplacedDetectorImageFilter.h"
#  include "RTKExport.h"

#  include <itkCudaImage.h>
#  include <itkCudaInPlaceImageFilter.h>

namespace rtk
{

/** \class CudaDisplacedDetectorImageFilter
 * \brief Cuda version of rtk::DisplacedDetectorImageFilter.
 *
 * Cuda version of rtk::DisplacedDetectorImageFilter.
 *
 * \see rtk::DisplacedDetectorImageFilter
 *
 * \test rtkdisplaceddetectortest.cxx, rtkdisplaceddetectorcompcudatest.cxx
 *
 * \author peter
 *
 * \ingroup RTK
 * \version 0.1
 */
class RTK_EXPORT CudaDisplacedDetectorImageFilter
  : public itk::CudaInPlaceImageFilter<itk::CudaImage<float, 3>,
                                       itk::CudaImage<float, 3>,
                                       rtk::DisplacedDetectorImageFilter<itk::CudaImage<float, 3>>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaDisplacedDetectorImageFilter);

  /** Convenience type alias **/
  using ImageType = itk::CudaImage<float, 3>;
  using CPUWeightFilterType = rtk::DisplacedDetectorImageFilter<ImageType>;

  /** Standard class type alias. */
  using Self = CudaDisplacedDetectorImageFilter;
  using Superclass = itk::CudaInPlaceImageFilter<ImageType, ImageType, CPUWeightFilterType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(CudaDisplacedDetectorImageFilter);

protected:
  /** Standard constructor **/
  CudaDisplacedDetectorImageFilter();
  /** Destructor **/
  virtual ~CudaDisplacedDetectorImageFilter();

  virtual void
  GPUGenerateData();
};

} // namespace rtk

#endif // end conditional definition of the class

#endif // rtkCudaDisplacedDetectorImageFilter_h
