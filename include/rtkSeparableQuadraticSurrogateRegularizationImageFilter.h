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

#ifndef rtkSeparableQuadraticSurrogateRegularizationImageFilter_h
#define rtkSeparableQuadraticSurrogateRegularizationImageFilter_h

#include <itkConstNeighborhoodIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImageToImageFilter.h>

namespace rtk
{

/** \class SeparableQuadraticSurrogateRegularizationImageFilter
 * \brief For one-step inversion of spectral CT data by the method Mechlem2017, computes regularization term's first and second derivatives
 *
 * In SQS-based methods (Long2014, Weidinger2016, Mechlem2017), the regularization term is typically
 * the sum of the "absolute values" of the differences between each pixel and its neighbours. Instead of
 * the actual absolute value function, which can't be differentiated in zero, smooth approximations are used,
 * e.g. the Huber function. Here, we employ another approximation, described in "Bayesian Reconstructions
 * From Emission Tomography Data Using a Modified EM Algorithm", by Peter J. Green, IEEE TMI 1990.
 * With respect to Huber, it has the advantage of not requiring a cut-off parameter.
 *
 * \ingroup IntensityImageFilters
 */

template<typename TImage>
class SeparableQuadraticSurrogateRegularizationImageFilter : public itk::ImageToImageFilter<TImage, TImage>
{
public:

  /** Standard class typedefs. */
  typedef SeparableQuadraticSurrogateRegularizationImageFilter      Self;
  typedef itk::ImageToImageFilter< TImage, TImage>                  Superclass;
  typedef itk::SmartPointer< Self >                                 Pointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(SeparableQuadraticSurrogateRegularizationImageFilter, itk::ImageToImageFilter)

  /** Set/Get for the radius */
  itkSetMacro(Radius, typename TImage::RegionType::SizeType)
  itkGetMacro(Radius, typename TImage::RegionType::SizeType)

  /** Set/Get for the regularization weights */
  itkSetMacro(RegularizationWeights, typename TImage::PixelType)
  itkGetMacro(RegularizationWeights, typename TImage::PixelType)

protected:
  SeparableQuadraticSurrogateRegularizationImageFilter();
  virtual ~SeparableQuadraticSurrogateRegularizationImageFilter() ITK_OVERRIDE {}

  /** Creates the Outputs */
  itk::DataObject::Pointer MakeOutput(unsigned int idx);

  /** Does the real work. */
#if ITK_VERSION_MAJOR<5
  void ThreadedGenerateData(const typename TImage::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;
#else
  void DynamicThreadedGenerateData(const typename TImage::RegionType& outputRegionForThread) ITK_OVERRIDE;
#endif
  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  /** Derivatives of the absolute value approximation */
  typename TImage::PixelType GreenPriorFirstDerivative(typename TImage::PixelType pix);
  typename TImage::PixelType GreenPriorSecondDerivative(typename TImage::PixelType pix);

  /** Member variables */
  typename TImage::RegionType::SizeType                             m_Radius;
  typename itk::PixelTraits<typename TImage::PixelType>::ValueType  m_c1;
  typename itk::PixelTraits<typename TImage::PixelType>::ValueType  m_c2;
  typename TImage::PixelType                                        m_RegularizationWeights;

private:
  SeparableQuadraticSurrogateRegularizationImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSeparableQuadraticSurrogateRegularizationImageFilter.hxx"
#endif

#endif
