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

#ifndef rtkAttenuatedToExponentialCorrectionImageFilter_h
#define rtkAttenuatedToExponentialCorrectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkMacro.h"
#include <itkPixelTraits.h>
#include <cmath>
#include <vector>

namespace rtk
{
namespace Functor
{
/** \class InterpolationWeightMultiplicationAttToExp
 * \brief Function to check if pixel along is in the K region and to record the
 * step length.
 *
 * \author Simon Rit
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TCoordRepType, class TOutput = TInput>
class ITK_TEMPLATE_EXPORT InterpolationWeightMultiplicationAttToExp
{
public:
  InterpolationWeightMultiplicationAttToExp() = default;

  ~InterpolationWeightMultiplicationAttToExp() = default;
  bool
  operator!=(const InterpolationWeightMultiplicationAttToExp &) const
  {
    return false;
  }

  bool
  operator==(const InterpolationWeightMultiplicationAttToExp & other) const
  {
    return !(*this != other);
  }

  inline TOutput
  operator()(const ThreadIdType  threadId,
             const double        stepLengthInVoxel,
             const TCoordRepType weight,
             const TInput *      p,
             const int           i)
  {
    const TInput kRegion = (p + m_KRegionMinusAttenuationMapPtrDiff)[i];
    const int    bNotInKRegion = (kRegion == itk::NumericTraits<TInput>::ZeroValue());
    m_BeforeKRegion[threadId] = m_BeforeKRegion[threadId] * bNotInKRegion;
    m_LatestStepLength[threadId] = stepLengthInVoxel;
    return weight * p[i];
  }

  void
  SetKRegionMinusAttenuationMapPtrDiff(std::ptrdiff_t pd)
  {
    m_KRegionMinusAttenuationMapPtrDiff = pd;
  }

  int *
  GetBeforeKRegion()
  {
    return m_BeforeKRegion;
  }

  double *
  GetLatestStepLength()
  {
    return m_LatestStepLength;
  }

private:
  std::ptrdiff_t m_KRegionMinusAttenuationMapPtrDiff{};
  int            m_BeforeKRegion[itk::ITK_MAX_THREADS]{ 1 };
  double         m_LatestStepLength[itk::ITK_MAX_THREADS]{};
};

/** \class SumAttenuationForCorrection
 * \brief Function to calculate the correction factor from attenuation to
 * exponential Radon transform.
 *
 * \author Simon Rit
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TOutput>
class ITK_TEMPLATE_EXPORT SumAttenuationForCorrection
{
public:
  using VectorType = itk::Vector<double, 3>;

  SumAttenuationForCorrection() = default;

  ~SumAttenuationForCorrection() = default;
  bool
  operator!=(const SumAttenuationForCorrection &) const
  {
    return false;
  }

  bool
  operator==(const SumAttenuationForCorrection & other) const
  {
    return !(*this != other);
  }

  inline void
  operator()(const ThreadIdType threadId,
             TOutput &          sumValue,
             const TInput       volumeValue,
             const VectorType & itkNotUsed(stepInMM))
  {
    sumValue += static_cast<TOutput>(volumeValue) * m_BeforeKRegion[threadId];
    m_TraveledBeforeKRegion[threadId] += m_LatestStepLength[threadId] * m_BeforeKRegion[threadId];
  }

  double *
  GetTraveledBeforeKRegion()
  {
    return m_TraveledBeforeKRegion;
  }

  void
  SetBeforeKRegion(int * pixelInKRegion)
  {
    m_BeforeKRegion = pixelInKRegion;
  }

  void
  SetLatestStepLength(double * latestStepLength)
  {
    m_LatestStepLength = latestStepLength;
  }

private:
  int *    m_BeforeKRegion{};
  double   m_TraveledBeforeKRegion[itk::ITK_MAX_THREADS]{ 0. };
  double * m_LatestStepLength{};
};

/** \class ProjectedValueAccumulationAttToExp
 * \brief Function to calculate the correction factor from attenuation to
 * exponential Radon transform using the integral of the attenuation before the
 * K region and the length traveled in the attenuation image before the K
 * region.
 *
 * \author Simon Rit
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TOutput>
class ITK_TEMPLATE_EXPORT ProjectedValueAccumulationAttToExp
{
public:
  using VectorType = itk::Vector<double, 3>;

  ProjectedValueAccumulationAttToExp() = default;
  ~ProjectedValueAccumulationAttToExp() = default;
  bool
  operator!=(const ProjectedValueAccumulationAttToExp &) const
  {
    return false;
  }

  bool
  operator==(const ProjectedValueAccumulationAttToExp & other) const
  {
    return !(*this != other);
  }

  inline void
  operator()(const ThreadIdType threadId,
             const TInput &     input,
             TOutput &          output,
             const TOutput &    rayCastValue,
             const VectorType & stepInMM,
             const VectorType & itkNotUsed(pixel),
             const VectorType & pixelToSource,
             const VectorType & nearestPoint,
             const VectorType & itkNotUsed(farthestPoint))
  {
    // If we have not hit the K region, we set the correction factor to 0. as
    // there shouldn't be any emission outside the K region.
    if (m_BeforeKRegion[threadId])
    {
      output = 0.;
      m_TraveledBeforeKRegion[threadId] = 0.;
      return;
    }

    // Calculate tau, the distance from the entrance point of the K region to
    // the origin. We assume that the detector is orthogonal to the
    // pixelToSource line.
    VectorType originToNearest = nearestPoint - m_Origin;
    VectorType originToNearestInMM = itk::MakeVector(
      originToNearest[0] * m_Spacing[0], originToNearest[1] * m_Spacing[1], originToNearest[2] * m_Spacing[2]);
    VectorType nearestToKRegionInMM = m_TraveledBeforeKRegion[threadId] * stepInMM;
    VectorType originToKRegionInMM = originToNearestInMM + nearestToKRegionInMM;
    VectorType pixelToSourceInMM = itk::MakeVector(
      pixelToSource[0] * m_Spacing[0], pixelToSource[1] * m_Spacing[1], pixelToSource[2] * m_Spacing[2]);
    double tau = originToKRegionInMM * pixelToSourceInMM / -pixelToSourceInMM.GetNorm();
    output = input * std::exp(rayCastValue * stepInMM.GetNorm() + tau * m_Mu0);

    // Reinitialize for next ray
    m_BeforeKRegion[threadId] = 1;
    m_TraveledBeforeKRegion[threadId] = 0.;
  }

  void
  SetBeforeKRegion(int * pixelInKRegion)
  {
    m_BeforeKRegion = pixelInKRegion;
  }

  void
  SetTraveledBeforeKRegion(double * traveledBeforeKRegion)
  {
    m_TraveledBeforeKRegion = traveledBeforeKRegion;
  }

  void
  SetMu0(TOutput mu0)
  {
    m_Mu0 = mu0;
  }

  void
  SetOrigin(VectorType origin)
  {
    m_Origin = origin;
  }

  void
  SetSpacing(VectorType spacing)
  {
    m_Spacing = spacing;
  }

private:
  int *      m_BeforeKRegion{};
  double *   m_TraveledBeforeKRegion{};
  TOutput    m_Mu0{};
  VectorType m_Origin{};
  VectorType m_Spacing{};
};
} // end namespace Functor

/** \class AttenuatedToExponentialCorrectionImageFilter
 * \brief Converts input projections from the attenuation to exponential Radon transform
 *
 * The conversion is explained p47 of Natterer's book, "The mathematics of
 * computerized tomography", 1986. The conversion assumes that the emission is
 * contained in a K region (referred to as $\Omega$ in the book) of constant
 * attenuation $\mu_0$. The projector therefore uses a mask as third input with
 * the same voxel lattice as the attenuation map in the second input. $\mu_0$ is
 * calculated as the average in the K region.
 *
 * \test TODO
 *
 * \author Simon Rit
 *
 * \ingroup RTK Projector
 */

template <
  class TInputImage,
  class TOutputImage,
  class TInterpolationWeightMultiplication = Functor::InterpolationWeightMultiplicationAttToExp<
    typename TInputImage::PixelType,
    typename itk::PixelTraits<typename TInputImage::PixelType>::ValueType>,
  class TProjectedValueAccumulation =
    Functor::ProjectedValueAccumulationAttToExp<typename TInputImage::PixelType, typename TOutputImage::PixelType>,
  class TSumAlongRay =
    Functor::SumAttenuationForCorrection<typename TInputImage::PixelType, typename TOutputImage::PixelType>>
class ITK_TEMPLATE_EXPORT AttenuatedToExponentialCorrectionImageFilter
  : public JosephForwardProjectionImageFilter<TInputImage,
                                              TOutputImage,
                                              TInterpolationWeightMultiplication,
                                              TProjectedValueAccumulation,
                                              TSumAlongRay>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AttenuatedToExponentialCorrectionImageFilter);

  /** Standard class type alias. */
  using Self = AttenuatedToExponentialCorrectionImageFilter;
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
  using CoordRepType = double;
  using VectorType = itk::Vector<CoordRepType, TInputImage::ImageDimension>;

  /** ImageDimension constants */
  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
#ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(AttenuatedToExponentialCorrectionImageFilter);
#else
  itkTypeMacro(AttenuatedToExponentialCorrectionImageFilter, JosephForwardProjectionImageFilter);
#endif

protected:
  AttenuatedToExponentialCorrectionImageFilter();
  ~AttenuatedToExponentialCorrectionImageFilter() override = default;

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
#  include "rtkAttenuatedToExponentialCorrectionImageFilter.hxx"
#endif

#endif
