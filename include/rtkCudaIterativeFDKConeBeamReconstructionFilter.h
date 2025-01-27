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

#ifndef rtkCudaIterativeFDKConeBeamReconstructionFilter_h
#define rtkCudaIterativeFDKConeBeamReconstructionFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkIterativeFDKConeBeamReconstructionFilter.h"
#  include "rtkCudaFDKConeBeamReconstructionFilter.h"
#  include "rtkCudaDisplacedDetectorImageFilter.h"
#  include "rtkCudaParkerShortScanImageFilter.h"
#  include "rtkCudaConstantVolumeSource.h"
#  include "RTKExport.h"

namespace rtk
{

/** \class CudaIterativeFDKConeBeamReconstructionFilter
 * \brief Implements the iterative FDK algorithm using Cuda
 *
 * Replaces:
 * - FDKConeBeamReconstructionFilter with CudaFDKConeBeamReconstructionFilter
 *
 * Also take care to create the reconstructed volume on the GPU at the beginning and
 * transfers it at the end.
 *
 * \test rtkfdktest.cxx, rtkrampfiltertest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK ReconstructionAlgorithm CudaImageToImageFilter
 */
class RTK_EXPORT CudaIterativeFDKConeBeamReconstructionFilter
  : public itk::CudaImageToImageFilter<
      itk::CudaImage<float, 3>,
      itk::CudaImage<float, 3>,
      IterativeFDKConeBeamReconstructionFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>, float>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaIterativeFDKConeBeamReconstructionFilter);

  /** Standard class type alias. */
  using Self = CudaIterativeFDKConeBeamReconstructionFilter;
  using Superclass =
    IterativeFDKConeBeamReconstructionFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>, float>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Typedefs of subfilters which have been implemented with CUDA */
  using DisplacedDetectorFilterType = rtk::CudaDisplacedDetectorImageFilter;
  using ParkerFilterType = rtk::CudaParkerShortScanImageFilter;
  using FDKFilterType = rtk::CudaFDKConeBeamReconstructionFilter;
  using ConstantImageSourceType = rtk::CudaConstantVolumeSource;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(CudaIterativeFDKConeBeamReconstructionFilter);

protected:
  CudaIterativeFDKConeBeamReconstructionFilter();
  ~CudaIterativeFDKConeBeamReconstructionFilter() {}

  virtual void
  GPUGenerateData();

}; // end of class

} // end namespace rtk

#endif // end conditional definition of the class

#endif
