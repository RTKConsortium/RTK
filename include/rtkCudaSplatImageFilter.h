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

#ifndef rtkCudaSplatImageFilter_h
#define rtkCudaSplatImageFilter_h

#include "rtkConfiguration.h"
//Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#include "rtkSplatWithKnownWeightsImageFilter.h"
#include "itkCudaImage.h"
#include "itkCudaInPlaceImageFilter.h"
#include "rtkWin32Header.h"

namespace rtk
{

/** \class CudaSplatImageFilter
 * \brief Implements the SplatWithKnownWeightsImageFilter on GPU.
 *
 *
 *
 * \author Cyril Mory
 *
 * \ingroup RTK CudaImageToImageFilter
 */
class RTK_EXPORT CudaSplatImageFilter :
    public itk::CudaInPlaceImageFilter< itk::CudaImage<float,4>, itk::CudaImage<float,4>,
  SplatWithKnownWeightsImageFilter< itk::CudaImage<float,4>, itk::CudaImage<float,3> > >

{
public:
  /** Standard class typedefs. */
  typedef rtk::CudaSplatImageFilter                             Self;
  typedef rtk::SplatWithKnownWeightsImageFilter< OutputImageType, InputImageType > Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(CudaSplatImageFilter, SplatWithKnownWeightsImageFilter)

protected:
  CudaSplatImageFilter();
  ~CudaSplatImageFilter(){
  }

  virtual void GPUGenerateData();

private:
  CudaSplatImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

}; // end of class

} // end namespace rtk

#endif //end conditional definition of the class

#endif
