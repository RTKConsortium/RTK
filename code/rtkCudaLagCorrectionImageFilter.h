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

#ifndef rtkCudaLagCorrectionImageFilter_h
#define rtkCudaLagCorrectionImageFilter_h

#include "rtkLagCorrectionImageFilter.h"
#include "rtkWin32Header.h"

#include <itkCudaImage.h>
#include <itkCudaInPlaceImageFilter.h>

#include "rtkConfiguration.h"

namespace rtk
{

/** \class CudaLagCorrectionImageFilter
 * \brief Cuda version of LagCorrectionImageFilter.
 *
 * Cuda version of LagCorrectionImageFilter.
 *
 * \see LagCorrectionImageFilter
 *
 * \author Sebastien Brousmiche
 */
class CudaLagCorrectionImageFilter :
  public  itk::CudaInPlaceImageFilter < itk::CudaImage<unsigned short, 3>, itk::CudaImage<unsigned short, 3>,
  LagCorrectionImageFilter < itk::CudaImage<unsigned short, 3>, 4> >
{
public:
  /** Convenience typedefs **/
  typedef itk::CudaImage<unsigned short, 3>                    ImageType;
  typedef LagCorrectionImageFilter <ImageType, 4>              CPULagFilterType;

  /** Standard class typedefs. */
  typedef CudaLagCorrectionImageFilter                                        Self;
  typedef itk::CudaInPlaceImageFilter<ImageType, ImageType, CPULagFilterType> Superclass;
  typedef itk::SmartPointer<Self>                                             Pointer;
  typedef itk::SmartPointer<const Self>                                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(Self, Superclass);

protected:
  /** Standard constructor **/
  rtkcuda_EXPORT CudaLagCorrectionImageFilter();
  /** Destructor **/
  virtual ~CudaLagCorrectionImageFilter();

  virtual void GPUGenerateData();

private:
  /** purposely not implemented **/
  CudaLagCorrectionImageFilter(const Self&);
  /** purposely not implemented **/
  void operator=(const Self&);
};

}

#endif // rtkCudaLagCorrectionImageFilter_h
