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

#ifndef __rtkAmsterdamShroudImageFilter_h
#define __rtkAmsterdamShroudImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkRecursiveGaussianImageFilter.h>
#if ITK_VERSION_MAJOR <= 3
#  include <itkMultiplyByConstantImageFilter.h>
#else
#  include <itkMultiplyImageFilter.h>
#endif
#include <itkThresholdImageFilter.h>
#include <itkSumProjectionImageFilter.h>
#include <itkConvolutionImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkPermuteAxesImageFilter.h>

namespace rtk
{

/** \class AmsterdamShroudImageFilter
 * \brief Compute the Amsterdam shroud image for respiratory signal extraction.
 *
 * The Amsterdam shroud is an image that is used to extract a respiratory
 * signal from cone-beam projection images. The Y-axis is time and the X-axis
 * is the cranio-caudal position. More information is available in
 * [Zijp, ICCR, 2004], [Sonke, Med Phys, 2005] and [Rit, IJROBP, 2012].
 *
 * \test rtkamsterdamshroudtest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template<class TInputImage>
class ITK_EXPORT AmsterdamShroudImageFilter :
  public itk::ImageToImageFilter<TInputImage, itk::Image<double, TInputImage::ImageDimension-1> >
{
public:
  /** Standard class typedefs. */
  typedef itk::Image<double, TInputImage::ImageDimension-1>     TOutputImage;
  typedef AmsterdamShroudImageFilter                            Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage>    Superclass;
  typedef itk::SmartPointer<Self>                               Pointer;
  typedef itk::SmartPointer<const Self>                         ConstPointer;

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(AmsterdamShroudImageFilter, itk::ImageToImageFilter);
protected:
  AmsterdamShroudImageFilter();
  ~AmsterdamShroudImageFilter(){}

  void GenerateOutputInformation();
  void GenerateInputRequestedRegion();

  /** Single-threaded version of GenerateData.  This filter delegates
   * to other filters. */
  void GenerateData();

private:
  AmsterdamShroudImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);             //purposely not implemented

  typedef itk::RecursiveGaussianImageFilter< TInputImage, TInputImage >          DerivativeType;
#if ITK_VERSION_MAJOR <= 3
  typedef itk::MultiplyByConstantImageFilter< TInputImage, double, TInputImage > NegativeType;
#else
  typedef itk::MultiplyImageFilter< TInputImage, TInputImage, TInputImage >      NegativeType;
#endif
  typedef itk::ThresholdImageFilter< TInputImage >                               ThresholdType;
  typedef itk::SumProjectionImageFilter< TInputImage, TOutputImage >             SumType;
  typedef itk::ConvolutionImageFilter< TOutputImage, TOutputImage >              ConvolutionType;
  typedef itk::SubtractImageFilter< TOutputImage, TOutputImage >                 SubtractType;
  typedef itk::PermuteAxesImageFilter< TOutputImage >                            PermuteType;

  typename DerivativeType::Pointer  m_DerivativeFilter;
  typename NegativeType::Pointer    m_NegativeFilter;
  typename ThresholdType::Pointer   m_ThresholdFilter;
  typename SumType::Pointer         m_SumFilter;
  typename ConvolutionType::Pointer m_ConvolutionFilter;
  typename SubtractType::Pointer    m_SubtractFilter;
  typename PermuteType::Pointer     m_PermuteFilter;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkAmsterdamShroudImageFilter.txx"
#endif

#endif
