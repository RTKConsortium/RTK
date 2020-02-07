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

#ifndef rtkJosephBackAttenuatedProjectionImageFilter_h
#define rtkJosephBackAttenuatedProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{
namespace Functor
{
/** \class InterpolationWeightMultiplicationAttenuatedBackProjection
 * \brief Function to multiply the interpolation weights with the projected
 * volume values and attenuation map.
 *
 * \author Antoine Robert
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TCoordRepType, class TOutput = TInput>
class InterpolationWeightMultiplicationAttenuatedBackProjection
{
public:
  InterpolationWeightMultiplicationAttenuatedBackProjection() { m_AttenuationPixel = 0; }

  ~InterpolationWeightMultiplicationAttenuatedBackProjection() = default;
  bool
  operator!=(const InterpolationWeightMultiplicationAttenuatedBackProjection &) const
  {
    return false;
  }

  bool
  operator==(const InterpolationWeightMultiplicationAttenuatedBackProjection & other) const
  {
    return !(*this != other);
  }

  inline TOutput
  operator()(const double stepLengthInVoxel, const TCoordRepType weight, const TInput * p, const int i)
  {
    const double w = weight * stepLengthInVoxel;

    m_AttenuationPixel += w * (p + m_AttenuationMinusEmissionMapsPtrDiff)[i];
    return w * (p + m_AttenuationMinusEmissionMapsPtrDiff)[i];
  }

  void
  SetAttenuationMinusEmissionMapsPtrDiff(std::ptrdiff_t pd)
  {
    m_AttenuationMinusEmissionMapsPtrDiff = pd;
  }
  TOutput *
  GetAttenuationPixel()
  {
    return &m_AttenuationPixel;
  }

private:
  std::ptrdiff_t m_AttenuationMinusEmissionMapsPtrDiff;
  TInput         m_AttenuationPixel;
};

/** \class ComputeAttenuationCorrectionBackProjection
 * \brief Function to compute the attenuation correction on the projection.
 *
 * \author Antoine Robert
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TOutput>
class ComputeAttenuationCorrectionBackProjection
{
public:
  using VectorType = itk::Vector<double, 3>;

  ComputeAttenuationCorrectionBackProjection() { m_ex1 = 1; }

  ~ComputeAttenuationCorrectionBackProjection() = default;
  bool
  operator!=(const ComputeAttenuationCorrectionBackProjection &) const
  {
    return false;
  }

  bool
  operator==(const ComputeAttenuationCorrectionBackProjection & other) const
  {
    return !(*this != other);
  }

  inline TOutput
  operator()(const TInput rayValue, const TInput attenuationRay, const VectorType & stepInMM, bool & isNewRay)
  {
    if (isNewRay)
    {
      m_ex1 = 1;
      isNewRay = false;
    }
    TInput ex2 = exp(-attenuationRay * stepInMM.GetNorm());
    TInput wf;
    if (*m_AttenuationPixel > 0)
    {
      wf = (m_ex1 - ex2) / *m_AttenuationPixel;
    }
    else
    {
      wf = m_ex1 * stepInMM.GetNorm();
    }

    m_ex1 = ex2;
    *m_AttenuationPixel = 0;
    return wf * rayValue;
  }

  void
  SetAttenuationPixel(TInput * attenuationPixel)
  {
    m_AttenuationPixel = attenuationPixel;
  }

private:
  TInput   m_ex1;
  TInput * m_AttenuationPixel;
};

/** \class SplatWeightMultiplicationAttenuated
 * \brief Function to multiply the interpolation weights with the projection
 * values.
 *
 * \author Cyril Mory
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TCoordRepType, class TOutput = TCoordRepType>
class SplatWeightMultiplicationAttenuated
{
public:
  SplatWeightMultiplicationAttenuated() = default;
  ~SplatWeightMultiplicationAttenuated() = default;
  bool
  operator!=(const SplatWeightMultiplicationAttenuated &) const
  {
    return false;
  }

  bool
  operator==(const SplatWeightMultiplicationAttenuated & other) const
  {
    return !(*this != other);
  }

  inline void
  operator()(const TInput &      rayValue,
             TOutput &           output,
             const double        stepLengthInVoxel,
             const double        itkNotUsed(voxelSize),
             const TCoordRepType weight) const
  {
    output += rayValue * weight * stepLengthInVoxel;
  }
};
} // end namespace Functor

/** \class JosephBackAttenuatedProjectionImageFilter
 * \brief Attenuated Joseph back projection.
 *
 * Performs a attenuated back projection, i.e. smearing of ray value along its path,
 * using [Joseph, IEEE TMI, 1982] and [Gullberg, Phys. Med. Biol., 1985]. The back projector is the adjoint operator of
 * the forward attenuated projector
 *
 * \test rtkbackprojectiontest.cxx
 *
 * \author Antoine Robert
 *
 * \ingroup RTK Projector
 */

template <
  class TInputImage,
  class TOutputImage,
  class TInterpolationWeightMultiplication = Functor::InterpolationWeightMultiplicationAttenuatedBackProjection<
    typename TInputImage::PixelType,
    typename itk::PixelTraits<typename TInputImage::PixelType>::ValueType>,
  class TSplatWeightMultiplication = Functor::
    SplatWeightMultiplicationAttenuated<typename TInputImage::PixelType, double, typename TOutputImage::PixelType>,
  class TSumAlongRay = Functor::ComputeAttenuationCorrectionBackProjection<typename TInputImage::PixelType,
                                                                           typename TOutputImage::PixelType>>
class ITK_EXPORT JosephBackAttenuatedProjectionImageFilter
  : public JosephBackProjectionImageFilter<TInputImage,
                                           TOutputImage,
                                           TInterpolationWeightMultiplication,
                                           TSplatWeightMultiplication,
                                           TSumAlongRay>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(JosephBackAttenuatedProjectionImageFilter);

  /** Standard class type alias. */
  using Self = JosephBackAttenuatedProjectionImageFilter;
  using Superclass = JosephBackProjectionImageFilter<TInputImage,
                                                     TOutputImage,
                                                     TInterpolationWeightMultiplication,
                                                     TSplatWeightMultiplication,
                                                     TSumAlongRay>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using CoordRepType = double;
  using VectorType = itk::Vector<CoordRepType, TInputImage::ImageDimension>;
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  using GeometryPointer = typename GeometryType::Pointer;

  /** ImageDimension constants */
  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(JosephBackAttenuatedProjectionImageFilter, JosephBackProjectionImageFilter);

protected:
  JosephBackAttenuatedProjectionImageFilter();
  ~JosephBackAttenuatedProjectionImageFilter() override = default;

  /** Apply changes to the input image requested region. */
  void
  GenerateInputRequestedRegion() override;

  /** Only the last two inputs should be in the same space so we need
   * to overwrite the method. */
  void
  VerifyInputInformation() const override;

  void
  GenerateData() override;

  void
  Init();
};
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkJosephBackAttenuatedProjectionImageFilter.hxx"
#endif

#endif
