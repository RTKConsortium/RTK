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
#include <algorithm>
#include <array>
#include <cmath>
#include <itkPixelTraits.h>

namespace rtk
{

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

template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT JosephForwardAttenuatedProjectionImageFilter
  : public JosephForwardProjectionImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(JosephForwardAttenuatedProjectionImageFilter);

  /** Standard class type alias. */
  using Self = JosephForwardAttenuatedProjectionImageFilter;
  using Superclass = JosephForwardProjectionImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using PointType = typename TInputImage::PointType;
  using CoordinateType = double;
  using VectorType = itk::Vector<CoordinateType, TInputImage::ImageDimension>;
  using WeightCoordinateType = typename itk::PixelTraits<InputPixelType>::ValueType;

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

  /** Stride between the per-thread values, i.e., values of thread threadId
   * are at index threadId * ThreadStride. Each value is on its own 64-byte
   * cache line to avoid false sharing between threads. */
  static constexpr std::size_t ThreadStride = std::max<std::size_t>(1, 64 / sizeof(InputPixelType));

  std::ptrdiff_t                                                  m_AttenuationMinusEmissionMapsPtrDiff;
  std::array<InputPixelType, itk::ITK_MAX_THREADS * ThreadStride> m_AttenuationRay;
  std::array<InputPixelType, itk::ITK_MAX_THREADS * ThreadStride> m_AttenuationPixel;
  std::array<InputPixelType, itk::ITK_MAX_THREADS * ThreadStride> m_Ex1;

  void
  ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId) override;

#if !defined(ITK_WRAPPING_PARSER)
  // Lambdas of the attenuated projector. Their types are used to detect that
  // they have not been replaced. They access the state of the filter through
  // their filter argument, which is generic so that the class is complete
  // when their bodies are instantiated.

  /** Function to multiply the interpolation weights with the projected
   * volume values and attenuation map. */
  static constexpr auto AttenuatedInterpolationWeightMultiplication = [](const ThreadIdType         threadId,
                                                                         auto &                     filter,
                                                                         double                     stepLengthInVoxel,
                                                                         const WeightCoordinateType weight,
                                                                         const InputPixelType *     p,
                                                                         const ptrdiff_t i) -> OutputPixelType {
    Self &            self = static_cast<Self &>(filter);
    const std::size_t t = threadId * ThreadStride;
    const double      w = weight * stepLengthInVoxel;

    self.m_AttenuationRay[t] += w * (p + self.m_AttenuationMinusEmissionMapsPtrDiff)[i];
    self.m_AttenuationPixel[t] += w * (p + self.m_AttenuationMinusEmissionMapsPtrDiff)[i];
    return weight * p[i];
  };
  using AttenuatedInterpolationWeightMultiplicationType =
    std::remove_const_t<decltype(AttenuatedInterpolationWeightMultiplication)>;

  /** Function to compute the attenuation correction on the projection. */
  static constexpr auto AttenuatedSumAlongRay = [](const ThreadIdType   threadId,
                                                   auto &               filter,
                                                   OutputPixelType &    sumValue,
                                                   const InputPixelType volumeValue,
                                                   const VectorType &   stepInMM) {
    Self &            self = static_cast<Self &>(filter);
    const std::size_t t = threadId * ThreadStride;
    InputPixelType    ex2 = exp(-1. * self.m_AttenuationRay[t] * stepInMM.GetNorm());
    InputPixelType    wf;

    if (self.m_AttenuationPixel[t] > 0)
    {
      wf = (self.m_Ex1[t] - ex2) / self.m_AttenuationPixel[t];
    }
    else
    {
      wf = self.m_Ex1[t] * stepInMM.GetNorm();
    }

    self.m_Ex1[t] = ex2;
    self.m_AttenuationPixel[t] = 0;
    sumValue += wf * volumeValue;
  };
  using AttenuatedSumAlongRayType = std::remove_const_t<decltype(AttenuatedSumAlongRay)>;

  /** Function to accumulate the ray casting on the projection. */
  static constexpr auto AttenuatedProjectedValueAccumulation = [](const ThreadIdType      threadId,
                                                                  auto &                  filter,
                                                                  const InputPixelType &  input,
                                                                  OutputPixelType &       output,
                                                                  const OutputPixelType & rayCastValue,
                                                                  const VectorType &      itkNotUsed(stepInMM),
                                                                  const PointType &       itkNotUsed(source),
                                                                  const VectorType &      itkNotUsed(sourceToPixel),
                                                                  const PointType &       itkNotUsed(nearestPoint),
                                                                  const PointType &       itkNotUsed(farthestPoint)) {
    Self &            self = static_cast<Self &>(filter);
    const std::size_t t = threadId * ThreadStride;
    output = input + rayCastValue;
    self.m_AttenuationRay[t] = 0;
    self.m_Ex1[t] = 1;
  };
  using AttenuatedProjectedValueAccumulationType = std::remove_const_t<decltype(AttenuatedProjectedValueAccumulation)>;
#endif
};
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkJosephForwardAttenuatedProjectionImageFilter.hxx"
#endif

#endif
