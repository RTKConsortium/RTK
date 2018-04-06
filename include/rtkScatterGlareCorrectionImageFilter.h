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

#ifndef rtkScatterGlareCorrectionImageFilter_h
#define rtkScatterGlareCorrectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkFFTConvolutionImageFilter.h"

namespace rtk
{

/** \class ScatterGlareCorrectionImageFilter
 * \brief Implements the scatter glare correction as described in [Poludniowski, PMB 2011].
 *
 * The filter code is based on FFTConvolutionImageFilter by Gaetan Lehmann
 * (see http://hdl.handle.net/10380/3154)
 *
 * \test rtkscatterglaretest.cxx
 *
 * \author Sebastien Brousmiche
 *
 * \ingroup ImageToImageFilter
 */

template<class TInputImage, class TOutputImage=TInputImage, class TFFTPrecision=double>
class ITK_EXPORT ScatterGlareCorrectionImageFilter :
    public rtk::FFTConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
{
public:
  /** Standard class typedefs. */
  typedef ScatterGlareCorrectionImageFilter                  Self;
  typedef rtk::FFTConvolutionImageFilter< TInputImage,
                                          TOutputImage,
                                          TFFTPrecision>     Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                                       InputImageType;
  typedef TOutputImage                                      OutputImageType;
  typedef TFFTPrecision                                     FFTPrecisionType;
  typedef typename InputImageType::IndexType                IndexType;
  typedef typename InputImageType::SizeType                 SizeType;

  typedef typename Superclass::FFTInputImageType            FFTInputImageType;
  typedef typename FFTInputImageType::Pointer               FFTInputImagePointer;
  typedef typename Superclass::FFTOutputImageType           FFTOutputImageType;
  typedef typename FFTOutputImageType::Pointer              FFTOutputImagePointer;

  typedef typename std::vector<float>                       CoefficientVectorType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ScatterGlareCorrectionImageFilter, FFTConvolutionImageFilter);

  itkGetConstMacro(Coefficients, CoefficientVectorType);
  virtual void SetCoefficients(const CoefficientVectorType coefficients)
    {
    if (this->m_Coefficients != coefficients)
      {
      this->m_Coefficients = coefficients;
      this->Modified();
      }
    }

protected:
  ScatterGlareCorrectionImageFilter();
  ~ScatterGlareCorrectionImageFilter() {}

  /** Create the deconvolution kernel
  */
  void UpdateFFTConvolutionKernel(const SizeType size) ITK_OVERRIDE;

private:
  ScatterGlareCorrectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);     //purposely not implemented

  CoefficientVectorType m_Coefficients;
  CoefficientVectorType m_PreviousCoefficients;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkScatterGlareCorrectionImageFilter.hxx"
#endif

#endif
