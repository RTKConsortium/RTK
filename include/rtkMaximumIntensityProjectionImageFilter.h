/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkMaximumIntensityProjectionImageFilter_h
#define rtkMaximumIntensityProjectionImageFilter_h

#include "rtkJosephForwardProjectionImageFilter.h"

namespace rtk
{
namespace Functor
{

/** \class MaximumIntensityAlongRay
 * \brief Function to compute the maximum intensity (MIP) value along the ray projection.
 *
 * \author Mikhail Polkovnikov
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TOutput>
class ITK_TEMPLATE_EXPORT MaximumIntensityAlongRay
{
public:
  using VectorType = itk::Vector<double, 3>;

  MaximumIntensityAlongRay() = default;
  ~MaximumIntensityAlongRay() = default;
  bool
  operator!=(const MaximumIntensityAlongRay &) const
  {
    return false;
  }
  bool
  operator==(const MaximumIntensityAlongRay & other) const
  {
    return !(*this != other);
  }

  inline void
  operator()(const ThreadIdType itkNotUsed(threadId),
             TOutput &          mipValue,
             const TInput       volumeValue,
             const VectorType & itkNotUsed(stepInMM))
  {
    TOutput tmp = static_cast<TOutput>(volumeValue);
    if (tmp > mipValue)
    {
      mipValue = tmp;
    }
  }
};

/** \class MaximumIntensityProjectedValueAccumulation
 * \brief Function to calculate maximum intensity step along the ray projection.
 *
 * \author Mikhail Polkovnikov
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TOutput>
class ITK_TEMPLATE_EXPORT MaximumIntensityProjectedValueAccumulation
{
public:
  using VectorType = itk::Vector<double, 3>;
  using PointType = itk::Point<double, 3>;

  bool
  operator!=(const MaximumIntensityProjectedValueAccumulation &) const
  {
    return false;
  }
  bool
  operator==(const MaximumIntensityProjectedValueAccumulation & other) const
  {
    return !(*this != other);
  }

  inline void
  operator()(const ThreadIdType itkNotUsed(threadId),
             const TInput &     input,
             TOutput &          output,
             const TOutput &    rayCastValue,
             const VectorType & stepInMM,
             const PointType &  itkNotUsed(source),
             const VectorType & itkNotUsed(sourceToPixel),
             const PointType &  itkNotUsed(nearestPoint),
             const PointType &  itkNotUsed(farthestPoint)) const
  {
    TOutput tmp = static_cast<TOutput>(input);
    if (tmp < rayCastValue)
    {
      tmp = rayCastValue;
    }
    output = tmp * stepInMM.GetNorm();
  }
};

} // end namespace Functor


/** \class MaximumIntensityProjectionImageFilter
 * \brief MIP filter.
 *
 * Performs a MIP forward projection, i.e. calculation of a maximum intensity
 * step along the x-ray line.
 *
 * \author Mikhail Polkovnikov
 *
 * \ingroup RTK Projector
 */

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication = Functor::InterpolationWeightMultiplication<
            typename TInputImage::PixelType,
            typename itk::PixelTraits<typename TInputImage::PixelType>::ValueType>,
          class TProjectedValueAccumulation =
            Functor::MaximumIntensityProjectedValueAccumulation<typename TInputImage::PixelType,
                                                                typename TOutputImage::PixelType>,
          class TSumAlongRay =
            Functor::MaximumIntensityAlongRay<typename TInputImage::PixelType, typename TOutputImage::PixelType>>
class ITK_TEMPLATE_EXPORT MaximumIntensityProjectionImageFilter
  : public JosephForwardProjectionImageFilter<TInputImage,
                                              TOutputImage,
                                              TInterpolationWeightMultiplication,
                                              TProjectedValueAccumulation,
                                              TSumAlongRay>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MaximumIntensityProjectionImageFilter);

  /** Standard class type alias. */
  using Self = MaximumIntensityProjectionImageFilter;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(MaximumIntensityProjectionImageFilter);

protected:
  MaximumIntensityProjectionImageFilter() = default;
  ~MaximumIntensityProjectionImageFilter() override = default;
};
} // end namespace rtk

#endif
