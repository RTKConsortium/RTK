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

#ifndef __rtkSiddonForwardProjectionImageFilter_h
#define __rtkSiddonForwardProjectionImageFilter_h

#define EPSILON 0.00001
#include "rtkConfiguration.h"
#include "rtkForwardProjectionImageFilter.h"

namespace rtk
{
/** \class SiddonForwardProjectionImageFilter
 * \brief Siddon forward projection.
 *
 * It calculates digitally reconstructed radiograph from given input using the Siddon
 * ray casting algorithm (platimatch version).
 *
 * \author Marc Vila
 *
 * \ingroup Projector
 */

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication = Functor::InterpolationWeightMultiplication<typename TInputImage::PixelType, double>,
          class TProjectedValueAccumulation        = Functor::ProjectedValueAccumulation<typename TInputImage::PixelType, typename TOutputImage::PixelType>
          >
class ITK_EXPORT SiddonForwardProjectionImageFilter :
  public ForwardProjectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef SiddonForwardProjectionImageFilter                     Self;
  typedef ForwardProjectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                Pointer;
  typedef itk::SmartPointer<const Self>                          ConstPointer;
  typedef typename TInputImage::PixelType                        InputPixelType;
  typedef typename TOutputImage::PixelType                       OutputPixelType;
  typedef typename TOutputImage::RegionType                      OutputImageRegionType;
  typedef double                                                 CoordRepType;
  typedef itk::Vector<CoordRepType, TInputImage::ImageDimension> VectorType;
  typedef itk::Point<double, 3>                                  PointType;
  typedef itk::Size<2>                                           SizeType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SiddonForwardProjectionImageFilter, ForwardProjectionImageFilter);

  /** Get/Set the functor that is used to multiply each interpolation value with a volume value */
  TInterpolationWeightMultiplication &       GetInterpolationWeightMultiplication() { return m_InterpolationWeightMultiplication; }
  const TInterpolationWeightMultiplication & GetInterpolationWeightMultiplication() const { return m_InterpolationWeightMultiplication; }
  void SetInterpolationWeightMultiplication(const TInterpolationWeightMultiplication & _arg)
  {
    if ( m_InterpolationWeightMultiplication != _arg )
      {
      m_InterpolationWeightMultiplication = _arg;
      this->Modified();
      }
  }

  /** Get/Set the functor that is used to accumulate values in the projection image after the ray
   * casting has been performed. */
  TProjectedValueAccumulation &       GetProjectedValueAccumulation() { return m_ProjectedValueAccumulation; }
  const TProjectedValueAccumulation & GetProjectedValueAccumulation() const { return m_ProjectedValueAccumulation; }
  void SetProjectedValueAccumulation(const TProjectedValueAccumulation & _arg)
  {
    if ( m_ProjectedValueAccumulation != _arg )
      {
      m_ProjectedValueAccumulation = _arg;
      this->Modified();
      }
  }

protected:
  SiddonForwardProjectionImageFilter() {}
  virtual ~SiddonForwardProjectionImageFilter() {}

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  virtual void VerifyInputInformation() {}

private:
  SiddonForwardProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);    //purposely not implemented

  TInterpolationWeightMultiplication m_InterpolationWeightMultiplication;
  TProjectedValueAccumulation        m_ProjectedValueAccumulation;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSiddonForwardProjectionImageFilter.txx"
#endif

#endif
