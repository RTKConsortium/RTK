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

#ifndef rtkCudaFDKWeightProjectionFilter_h
#define rtkCudaFDKWeightProjectionFilter_h

#include "rtkFDKWeightProjectionFilter.h"
#include "rtkWin32Header.h"

#include <itkCudaImage.h>
#include <itkCudaInPlaceImageFilter.h>

namespace rtk
{

/** \class CudaFDKWeightProjectionFilter
 * \brief Cuda version of rtk::FDKWeightProjectionFilter.
 *
 * Cuda version of rtk::FDKWeightProjectionFilter.
 *
 * \see rtk::FDKWeightProjectionFilter
 *
 * \test rtkfdktest.cxx, rtkrampfiltertest.cxx, rtkdisplaceddetectortest.cxx,
 * rtkshortscantest.cxx, rtkfdkprojweightcompcudatest.cxx
 *
 * \author peter
 * \version 0.1
 */
class CudaFDKWeightProjectionFilter :
    public  itk::CudaInPlaceImageFilter<itk::CudaImage<float,3>, itk::CudaImage<float,3>,
            rtk::FDKWeightProjectionFilter<itk::CudaImage<float, 3> > >
{
public:
  /** Convenience typedefs **/
  typedef itk::CudaImage<float,3>                    ImageType;
  typedef rtk::FDKWeightProjectionFilter<ImageType>  CPUWeightFilterType;

  /** Standard class typedefs. */
  typedef CudaFDKWeightProjectionFilter                         Self;
  typedef itk::CudaInPlaceImageFilter<ImageType, ImageType,
    CPUWeightFilterType>                                        Superclass;
  typedef itk::SmartPointer<Self>                               Pointer;
  typedef itk::SmartPointer<const Self>                         ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(Self, Superclass);

protected:
  /** Standard constructor **/
  rtkcuda_EXPORT CudaFDKWeightProjectionFilter();
  /** Destructor **/
  virtual ~CudaFDKWeightProjectionFilter();

  virtual void GPUGenerateData();

private:
  /** purposely not implemented **/
  CudaFDKWeightProjectionFilter(const Self&);
  /** purposely not implemented **/
  void operator=(const Self&);

}; // end of class

} // end namespace rtk

#endif // rtkCudaFDKWeightProjectionFilter_h
