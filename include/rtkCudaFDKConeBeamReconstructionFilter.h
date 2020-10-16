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

#ifndef rtkCudaFDKConeBeamReconstructionFilter_h
#define rtkCudaFDKConeBeamReconstructionFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkFDKConeBeamReconstructionFilter.h"
#  include "rtkCudaFDKWeightProjectionFilter.h"
#  include "rtkCudaFFTRampImageFilter.h"
#  include "rtkCudaFDKBackProjectionImageFilter.h"
#  include "RTKExport.h"

namespace rtk
{

/** \class CudaFDKConeBeamReconstructionFilter
 * \brief Implements [Feldkamp, Davis, Kress, 1984] algorithm using Cuda
 *
 * Replaces ramp filter and backprojection in FDKConeBeamReconstructionFilter
 * with CudaFFTRampImageFilter and CudaFDKBackProjectionImageFilter.
 * Also take care to create the reconstructed volume on the GPU at the beginning and
 * transfers it at the end.
 *
 * \test rtkfdktest.cxx, rtkrampfiltertest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK ReconstructionAlgorithm CudaImageToImageFilter
 */
class RTK_EXPORT CudaFDKConeBeamReconstructionFilter
  : public itk::CudaInPlaceImageFilter<
      itk::CudaImage<float, 3>,
      itk::CudaImage<float, 3>,
      FDKConeBeamReconstructionFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>, float>>
{
public:
#  if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(CudaFDKConeBeamReconstructionFilter);
#  else
  ITK_DISALLOW_COPY_AND_MOVE(CudaFDKConeBeamReconstructionFilter);
#  endif

  /** Standard class type alias. */
  using Self = CudaFDKConeBeamReconstructionFilter;
  using Superclass = FDKConeBeamReconstructionFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>, float>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Typedefs of subfilters which have been implemented with CUDA */
  using WeightFilterType = rtk::CudaFDKWeightProjectionFilter;
  using RampFilterType = rtk::CudaFFTRampImageFilter;
  using BackProjectionFilterType = rtk::CudaFDKBackProjectionImageFilter;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaFDKConeBeamReconstructionFilter, FDKConeBeamReconstructionFilter);

protected:
  CudaFDKConeBeamReconstructionFilter();
  ~CudaFDKConeBeamReconstructionFilter() {}

  virtual void
  GPUGenerateData();

}; // end of class

} // end namespace rtk

#endif // end conditional definition of the class

#endif
