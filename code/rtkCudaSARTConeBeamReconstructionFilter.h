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

#ifndef __rtkCudaSARTConeBeamReconstructionFilter_h
#define __rtkCudaSARTConeBeamReconstructionFilter_h

#include "rtkCudaBackProjectionImageFilter.h"
#include "rtkCudaForwardProjectionImageFilter.h"
#include "rtkSARTConeBeamReconstructionFilter.h"
#include "rtkWin32Header.h"

#include <itkExtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkTimeProbe.h>

#include <itkCudaImage.h>

namespace rtk
{

/** \class CudaSARTConeBeamReconstructionFilter
 * \brief Implements the Simultaneous Algebraic Reconstruction Technique [Andersen, 1984]
 *
 * CudaSARTConeBeamReconstructionFilter is a mini-pipeline filter which combines
 * the different steps of the SART cone-beam reconstruction, mainly:
 * - ExtractFilterType to work on one projection at a time
 * - ForwardProjectionImageFilter,
 * - SubtractImageFilter,
 * - BackProjectionImageFilter.
 * The input stack of projections is processed piece by piece (the size is
 * controlled with ProjectionSubsetSize) via the use of itk::ExtractImageFilter
 * to extract sub-stacks.
 *
 * \test rtksarttest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ReconstructionAlgorithm
 */
class ITK_EXPORT CudaSARTConeBeamReconstructionFilter :
  public rtk::SARTConeBeamReconstructionFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3> >
{
public:
  /** Standard class typedefs. */
  typedef CudaSARTConeBeamReconstructionFilter                                                  Self;
  typedef SARTConeBeamReconstructionFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3> >  Superclass;
  typedef itk::SmartPointer<Self>                                                               Pointer;
  typedef itk::SmartPointer<const Self>                                                         ConstPointer;

  /** Typedefs of each subfilter of this composite filter */
  typedef rtk::CudaForwardProjectionImageFilter                                          ForwardProjectionFilterType;
  typedef rtk::CudaBackProjectionImageFilter                                             BackProjectionFilterType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaSARTConeBeamReconstructionFilter, SARTConeBeamReconstructionFilter);

protected:
  rtkcuda_EXPORT CudaSARTConeBeamReconstructionFilter();
  ~CudaSARTConeBeamReconstructionFilter(){}

private:
  //purposely not implemented
  CudaSARTConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

}; // end of class

} // end namespace rtk

#endif
