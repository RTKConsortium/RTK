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

#ifndef rtkJosephBackAttenuatedProjectionImageFilter_h
#define rtkJosephBackAttenuatedProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{


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

template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT JosephBackAttenuatedProjectionImageFilter
  : public JosephBackProjectionImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(JosephBackAttenuatedProjectionImageFilter);

  /** Standard class type alias. */
  using Self = JosephBackAttenuatedProjectionImageFilter;
  using Superclass = JosephBackProjectionImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;
  using WeightCoordinateType = typename itk::PixelTraits<InputPixelType>::ValueType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using CoordinateType = double;
  using VectorType = itk::Vector<CoordinateType, TInputImage::ImageDimension>;
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  using GeometryPointer = typename GeometryType::Pointer;

  /** ImageDimension constants */
  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(JosephBackAttenuatedProjectionImageFilter);

protected:
  JosephBackAttenuatedProjectionImageFilter();
  ~JosephBackAttenuatedProjectionImageFilter() override = default;

  const InputPixelType * m_AttenuationBuffer{ nullptr };
  bool                   m_AttenuationSameLayout{ true };

  /** Apply changes to the input image requested region. */
  void
  GenerateInputRequestedRegion() override;

  /** Only the last two inputs should be in the same space so we need
   * to overwrite the method. */
  void
  VerifyInputInformation() const override;

  /** The attenuation map is sampled at offsets of the output buffer, which is
   * allocated here. */
  void
  AllocateOutputs() override;

  void
  Init();

#if !defined(ITK_WRAPPING_PARSER)
  /** Attenuation along the ray up to the current step and transmission at the previous step. */
  struct AttenuationRayState
  {
    OutputPixelType attenuationRay{ 0 };
    InputPixelType  ex1{ 1 };
  };

  /** Attenuation map values at the corner offsets of the volume buffer. */
  std::array<InputPixelType, 4>
  SampleAttenuation(const typename Superclass::CornerOffsetsType & offsets) const
  {
    std::array<InputPixelType, 4> samples;
    for (unsigned int k = 0; k < 4; ++k)
    {
      samples[k] = m_AttenuationSameLayout ? m_AttenuationBuffer[offsets[k]]
                                           : this->GetInput(2)->GetPixel(this->GetOutput()->ComputeIndex(offsets[k]));
    }
    return samples;
  }
#endif
};
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkJosephBackAttenuatedProjectionImageFilter.hxx"
#endif

#endif
