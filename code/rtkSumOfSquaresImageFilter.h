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

#ifndef rtkSumOfSquaresImageFilter_h
#define rtkSumOfSquaresImageFilter_h

#include <itkInPlaceImageFilter.h>
#include <itkVectorImage.h>

/** \class SumOfSquaresImageFilter
 * \brief Computes the sum of squared differences between two images
 *
 * Works on vector images too.
 *
 * \author Cyril Mory
 */
namespace rtk
{

template <class TOutputImage>
class ITK_EXPORT SumOfSquaresImageFilter :
  public itk::InPlaceImageFilter<TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef SumOfSquaresImageFilter                     Self;
  typedef itk::InPlaceImageFilter<TOutputImage>                  Superclass;
  typedef itk::SmartPointer<Self>                                Pointer;
  typedef itk::SmartPointer<const Self>                          ConstPointer;
  typedef typename TOutputImage::PixelType                       OutputPixelType;
  typedef typename TOutputImage::RegionType                      OutputImageRegionType;
  typedef typename TOutputImage::InternalPixelType               OutputInternalPixelType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SumOfSquaresImageFilter, itk::InPlaceImageFilter);

  /** Macro to get the SSD */
  itkGetMacro(SumOfSquares, OutputInternalPixelType);

protected:
  SumOfSquaresImageFilter();
  ~SumOfSquaresImageFilter() ITK_OVERRIDE {}

  void BeforeThreadedGenerateData();
  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType threadId ) ITK_OVERRIDE;
  void AfterThreadedGenerateData();

  OutputInternalPixelType m_SumOfSquares;
  std::vector<OutputInternalPixelType> m_VectorOfPartialSSs;

private:
  SumOfSquaresImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented
};

template <>
void
SumOfSquaresImageFilter<itk::VectorImage<float, 3>>
::ThreadedGenerateData(const itk::VectorImage<float, 3>::RegionType& outputRegionForThread, itk::ThreadIdType threadId);

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSumOfSquaresImageFilter.hxx"
#endif

#endif
