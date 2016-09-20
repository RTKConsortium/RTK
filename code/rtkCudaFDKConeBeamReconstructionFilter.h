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

#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkCudaFDKWeightProjectionFilter.h"
#include "rtkCudaFFTRampImageFilter.h"
#include "rtkCudaFDKBackProjectionImageFilter.h"
#include "rtkWin32Header.h"

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
 * \ingroup ReconstructionAlgorithm CudaImageToImageFilter
 */
class CudaFDKConeBeamReconstructionFilter :
  public itk::CudaInPlaceImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
  FDKConeBeamReconstructionFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>, float > >
{
public:
  /** Standard class typedefs. */
  typedef CudaFDKConeBeamReconstructionFilter                                                        Self;
  typedef FDKConeBeamReconstructionFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>, float > Superclass;
  typedef itk::SmartPointer<Self>                                                                    Pointer;
  typedef itk::SmartPointer<const Self>                                                              ConstPointer;

  /** Typedefs of subfilters which have been implemented with CUDA */
  typedef rtk::CudaFDKWeightProjectionFilter    WeightFilterType;
  typedef rtk::CudaFFTRampImageFilter           RampFilterType;
  typedef rtk::CudaFDKBackProjectionImageFilter BackProjectionFilterType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaFDKConeBeamReconstructionFilter, FDKConeBeamReconstructionFilter);

protected:
  rtkcuda_EXPORT CudaFDKConeBeamReconstructionFilter();
  ~CudaFDKConeBeamReconstructionFilter(){}

  virtual void GPUGenerateData();

private:
  //purposely not implemented
  CudaFDKConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);
}; // end of class

} // end namespace rtk

#endif
