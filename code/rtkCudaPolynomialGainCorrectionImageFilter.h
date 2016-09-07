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

#ifndef rtkCudaPolynomialGainCorrectionImageFilter_h
#define rtkCudaPolynomialGainCorrectionImageFilter_h

#include "rtkPolynomialGainCorrectionImageFilter.h"
#include "rtkWin32Header.h"

#include <itkCudaImage.h>
#include <itkCudaInPlaceImageFilter.h>

#include "rtkConfiguration.h"

namespace rtk
{

/** \class PolynomialGainCorrectionImageFilter
 * \brief Cuda version of PolynomialGainCorrectionImageFilter.
 *
 * Cuda version of PolynomialGainCorrectionImageFilter.
 *
 * \see PolynomialGainCorrectionImageFilter
 *
 * \author Sebastien Brousmiche
 */
class CudaPolynomialGainCorrectionImageFilter :
    public  itk::CudaInPlaceImageFilter < itk::CudaImage<unsigned short, 3>, itk::CudaImage<float, 3>,
    PolynomialGainCorrectionImageFilter <itk::CudaImage<unsigned short, 3>, itk::CudaImage<float, 3> > >
{
public:
  /** Convenience typedefs **/
  typedef itk::CudaImage<float, 3>                                                 ImageType;
  typedef PolynomialGainCorrectionImageFilter< itk::CudaImage<unsigned short, 3>,
                                               itk::CudaImage<float, 3> >          CPUPolyGainFilterType;

  /** Standard class typedefs. */
  typedef CudaPolynomialGainCorrectionImageFilter                                  Self;
  typedef itk::CudaInPlaceImageFilter<ImageType, ImageType, CPUPolyGainFilterType> Superclass;
  typedef itk::SmartPointer<Self>                                                  Pointer;
  typedef itk::SmartPointer<const Self>                                            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(Self, Superclass);

protected:
  /** Standard constructor **/
  rtkcuda_EXPORT CudaPolynomialGainCorrectionImageFilter();
  /** Destructor **/
  virtual ~CudaPolynomialGainCorrectionImageFilter();

  virtual void GPUGenerateData();

private:
  /** purposely not implemented **/
  CudaPolynomialGainCorrectionImageFilter(const Self&);
  /** purposely not implemented **/
  void operator=(const Self&);
};

}

#endif // rtkCudaPolynomialGainCorrectionImageFilter_h
