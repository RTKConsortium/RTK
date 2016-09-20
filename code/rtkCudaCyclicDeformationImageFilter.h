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

#ifndef rtkCudaCyclicDeformationImageFilter_h
#define rtkCudaCyclicDeformationImageFilter_h

#include "rtkCyclicDeformationImageFilter.h"
#include <itkCudaImageToImageFilter.h>
#include "rtkWin32Header.h"

namespace rtk
{

/** \class CudaCyclicDeformationImageFilter
 * \brief GPU version of the temporal DVF interpolator
 *
 * This filter implements linear interpolation along time
 * in a DVF, assuming that the motion is periodic
 *
 * \author Cyril Mory
 *
 * \ingroup CudaImageToImageFilter
 */

class ITK_EXPORT CudaCyclicDeformationImageFilter :
public itk::CudaImageToImageFilter< itk::CudaImage<itk::CovariantVector<float,3>, 4>,
                                    itk::CudaImage<itk::CovariantVector<float,3>, 3>,
                                    CyclicDeformationImageFilter< itk::CudaImage<itk::CovariantVector<float,3>, 3> > >
{
public:
  /** Standard class typedefs. */
  typedef rtk::CudaCyclicDeformationImageFilter                       Self;
  typedef itk::CudaImage<itk::CovariantVector<float,3>, 4>            InputImageType;
  typedef itk::CudaImage<itk::CovariantVector<float,3>, 3>            OutputImageType;
  typedef rtk::CyclicDeformationImageFilter< OutputImageType >        Superclass;
  typedef itk::SmartPointer<Self>                                     Pointer;
  typedef itk::SmartPointer<const Self>                               ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(CudaCyclicDeformationImageFilter, CyclicDeformationImageFilter)

protected:
  rtkcuda_EXPORT CudaCyclicDeformationImageFilter();
  ~CudaCyclicDeformationImageFilter(){
  }

  virtual void GPUGenerateData();

private:
  CudaCyclicDeformationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

}; // end of class

} // end namespace rtk

#endif
