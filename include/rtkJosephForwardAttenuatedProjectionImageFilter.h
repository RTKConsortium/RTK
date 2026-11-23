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

#ifndef rtkJosephForwardAttenuatedProjectionImageFilter_h
#define rtkJosephForwardAttenuatedProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkMacro.h"
#include <cmath>
#include <itkPixelTraits.h>
#include <algorithm>
#include <vector>

namespace rtk
{
namespace Functor
{
/** Stride between the per-thread values of the attenuated forward projector
 * functors, i.e., values of thread threadId are at index
 * threadId * JosephAttenuatedThreadStride<TInput>(). Each value is on its
 * own 64-byte cache line to avoid false sharing between threads. */
template <class TInput>
constexpr std::size_t
JosephAttenuatedThreadStride()
{
  return std::max<std::size_t>(1, 64 / sizeof(TInput));
}

/** \class InterpolationWeightMultiplicationAttenuated
 * \brief Function to multiply the interpolation weights with the projected
 * volume values and attenuation map.
 *
 * \author Antoine Robert
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TCoordinateType, class TOutput = TInput>
class ITK_TEMPLATE_EXPORT InterpolationWeightMultiplicationAttenuated
{
public:
  InterpolationWeightMultiplicationAttenuated()
  {
    std::fill(std::begin(m_AttenuationRay), std::end(m_AttenuationRay), 0);
    std::fill(std::begin(m_AttenuationPixel), std::end(m_AttenuationPixel), 0);
    std::fill(std::begin(m_Ex1), std::end(m_Ex1), 1);
  }

  ~InterpolationWeightMultiplicationAttenuated() = default;
  bool
  operator!=(const InterpolationWeightMultiplicationAttenuated &) const
  {
    return false;
  }

  bool
  operator==(const InterpolationWeightMultiplicationAttenuated & other) const
  {
    return !(*this != other);
  }

  TOutput
  operator()(const ThreadIdType    threadId,
             const double          stepLengthInVoxel,
             const TCoordinateType weight,
             const TInput *        p,
             const ptrdiff_t       i)
  {
    const double      w = weight * stepLengthInVoxel;
    const std::size_t t = threadId * Stride;

    m_AttenuationRay[t] += w * (p + m_AttenuationMinusEmissionMapsPtrDiff)[i];
    m_AttenuationPixel[t] += w * (p + m_AttenuationMinusEmissionMapsPtrDiff)[i];
    return weight * p[i];
  }

  void
  SetAttenuationMinusEmissionMapsPtrDiff(std::ptrdiff_t pd)
  {
    m_AttenuationMinusEmissionMapsPtrDiff = pd;
  }
  TOutput *
  GetAttenuationRay()
  {
    return m_AttenuationRay;
  }
  TOutput *
  GetAttenuationPixel()
  {
    return m_AttenuationPixel;
  }
  TOutput *
  GetEx1()
  {
    return m_Ex1;
  }

private:
  static constexpr std::size_t Stride = JosephAttenuatedThreadStride<TInput>();
  std::ptrdiff_t               m_AttenuationMinusEmissionMapsPtrDiff{};
  TInput                       m_AttenuationRay[itk::ITK_MAX_THREADS * Stride];
  TInput                       m_AttenuationPixel[itk::ITK_MAX_THREADS * Stride];
  TInput                       m_Ex1[itk::ITK_MAX_THREADS * Stride];
};

/** \class ComputeAttenuationCorrection
 * \brief Function to compute the attenuation correction on the projection.
 *
 * \author Antoine Robert
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TOutput>
class ITK_TEMPLATE_EXPORT ComputeAttenuationCorrection
{
public:
  using VectorType = itk::Vector<double, 3>;

  ComputeAttenuationCorrection() = default;
  ~ComputeAttenuationCorrection() = default;
  bool
  operator!=(const ComputeAttenuationCorrection &) const
  {
    return false;
  }

  bool
  operator==(const ComputeAttenuationCorrection & other) const
  {
    return !(*this != other);
  }

  void
  operator()(const ThreadIdType threadId, TOutput & sumValue, const TInput volumeValue, const VectorType & stepInMM)
  {
    const std::size_t t = threadId * JosephAttenuatedThreadStride<TInput>();
    TInput            ex2 = exp(-m_AttenuationRay[t] * stepInMM.GetNorm());
    TInput            wf;

    if (m_AttenuationPixel[t] > 0)
    {
      wf = (m_Ex1[t] - ex2) / m_AttenuationPixel[t];
    }
    else
    {
      wf = m_Ex1[t] * stepInMM.GetNorm();
    }

    m_Ex1[t] = ex2;
    m_AttenuationPixel[t] = 0;
    sumValue += wf * volumeValue;
  }

  void
  SetAttenuationRayVector(TInput * attenuationRayVector)
  {
    m_AttenuationRay = attenuationRayVector;
  }
  void
  SetAttenuationPixelVector(TInput * attenuationPixelVector)
  {
    m_AttenuationPixel = attenuationPixelVector;
  }
  void
  SetEx1(TInput * ex1)
  {
    m_Ex1 = ex1;
  }

private:
  TInput * m_AttenuationRay;
  TInput * m_AttenuationPixel;
  TInput * m_Ex1;
};

/** \class ProjectedValueAccumulationAttenuated
 * \brief Function to accumulate the ray casting on the projection.
 *
 * \author Antoine Robert
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TOutput>
class ITK_TEMPLATE_EXPORT ProjectedValueAccumulationAttenuated
{
public:
  using VectorType = itk::Vector<double, 3>;
  using PointType = itk::Point<double, 3>;

  ProjectedValueAccumulationAttenuated() = default;
  ~ProjectedValueAccumulationAttenuated() = default;
  bool
  operator!=(const ProjectedValueAccumulationAttenuated &) const
  {
    return false;
  }

  bool
  operator==(const ProjectedValueAccumulationAttenuated & other) const
  {
    return !(*this != other);
  }

  void
  operator()(const ThreadIdType threadId,
             const TInput &     input,
             TOutput &          output,
             const TOutput &    rayCastValue,
             const VectorType & /*stepInMM*/,
             const PointType &  itkNotUsed(source),
             const VectorType & itkNotUsed(sourceToPixel),
             const PointType &  itkNotUsed(nearestPoint),
             const PointType &  itkNotUsed(farthestPoint))
  {
    const std::size_t t = threadId * JosephAttenuatedThreadStride<TInput>();
    output = input + rayCastValue;
    m_Attenuation[t] = 0;
    m_Ex1[t] = 1;
  }

  void
  SetAttenuationVector(TInput * attenuationVector)
  {
    m_Attenuation = attenuationVector;
  }
  void
  SetEx1(TInput * ex1)
  {
    m_Ex1 = ex1;
  }

private:
  TInput * m_Attenuation;
  TInput * m_Ex1;
};
} // end namespace Functor

/** \class JosephForwardAttenuatedProjectionImageFilter
 * \brief Joseph forward projection.
 *
 * Performs a attenuated Joseph forward projection, i.e. accumulation along x-ray lines,
 * using [Joseph, IEEE TMI, 1982] and [Gullberg, Phys. Med. Biol., 1985]. The forward projector tests if the  detector
 * has been placed after the source and the volume. If the detector is in the volume
 * the ray tracing is performed only until that point.
 *
 * \test rtkforwardattenuatedprojectiontest.cxx
 *
 * \author Antoine Robert
 *
 * \ingroup RTK Projector
 */

template <
  class TInputImage,
  class TOutputImage,
  class TInterpolationWeightMultiplication = Functor::InterpolationWeightMultiplicationAttenuated<
    typename TInputImage::PixelType,
    typename itk::PixelTraits<typename TInputImage::PixelType>::ValueType>,
  class TProjectedValueAccumulation =
    Functor::ProjectedValueAccumulationAttenuated<typename TInputImage::PixelType, typename TOutputImage::PixelType>,
  class TSumAlongRay =
    Functor::ComputeAttenuationCorrection<typename TInputImage::PixelType, typename TOutputImage::PixelType>>
class ITK_TEMPLATE_EXPORT JosephForwardAttenuatedProjectionImageFilter
  : public JosephForwardProjectionImageFilter<TInputImage,
                                              TOutputImage,
                                              TInterpolationWeightMultiplication,
                                              TProjectedValueAccumulation,
                                              TSumAlongRay>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(JosephForwardAttenuatedProjectionImageFilter);

  /** Standard class type alias. */
  using Self = JosephForwardAttenuatedProjectionImageFilter;
  using Superclass = JosephForwardProjectionImageFilter<TInputImage,
                                                        TOutputImage,
                                                        TInterpolationWeightMultiplication,
                                                        TProjectedValueAccumulation,
                                                        TSumAlongRay>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using CoordinateType = double;
  using VectorType = itk::Vector<CoordinateType, TInputImage::ImageDimension>;

  /** ImageDimension constants */
  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(JosephForwardAttenuatedProjectionImageFilter);

protected:
  JosephForwardAttenuatedProjectionImageFilter();
  ~JosephForwardAttenuatedProjectionImageFilter() override = default;

  /** Apply changes to the input image requested region. */
  void
  GenerateInputRequestedRegion() override;

  void
  BeforeThreadedGenerateData() override;

  /** Only the last two inputs should be in the same space so we need
   * to overwrite the method. */
  void
  VerifyInputInformation() const override;
};
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkJosephForwardAttenuatedProjectionImageFilter.hxx"
#endif

#endif
