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

#ifndef rtkCudaIterativeFDKConeBeamReconstructionFilter_h
#define rtkCudaIterativeFDKConeBeamReconstructionFilter_h

#include "rtkIterativeFDKConeBeamReconstructionFilter.h"
#include "rtkCudaFDKConeBeamReconstructionFilter.h"
#include "rtkCudaDisplacedDetectorImageFilter.h"
#include "rtkCudaParkerShortScanImageFilter.h"
#include "rtkCudaConstantVolumeSource.h"
#include "rtkWin32Header.h"

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
 * \ingroup ReconstructionAlgorithm CudaImageToImageFilter
 */
class CudaIterativeFDKConeBeamReconstructionFilter :
  public itk::CudaInPlaceImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
  IterativeFDKConeBeamReconstructionFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>, float > >
{
public:
  /** Standard class typedefs. */
  typedef CudaIterativeFDKConeBeamReconstructionFilter                                                        Self;
  typedef IterativeFDKConeBeamReconstructionFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>, float > Superclass;
  typedef itk::SmartPointer<Self>                                                                             Pointer;
  typedef itk::SmartPointer<const Self>                                                                       ConstPointer;

  /** Typedefs of subfilters which have been implemented with CUDA */
  typedef rtk::CudaDisplacedDetectorImageFilter     DisplacedDetectorFilterType;
  typedef rtk::CudaParkerShortScanImageFilter       ParkerFilterType;
  typedef rtk::CudaFDKConeBeamReconstructionFilter  FDKFilterType;
  typedef rtk::CudaConstantVolumeSource             ConstantImageSourceType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaIterativeFDKConeBeamReconstructionFilter, IterativeFDKConeBeamReconstructionFilter);

protected:
  rtkcuda_EXPORT CudaIterativeFDKConeBeamReconstructionFilter();
  ~CudaIterativeFDKConeBeamReconstructionFilter(){}

  virtual void GPUGenerateData();

private:
  //purposely not implemented
  CudaIterativeFDKConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);
}; // end of class

} // end namespace rtk

#endif
