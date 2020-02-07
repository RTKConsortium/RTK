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

#ifndef rtkXRadRawToAttenuationImageFilter_h
#define rtkXRadRawToAttenuationImageFilter_h

#include <itkImageToImageFilter.h>
#include "rtkConfiguration.h"

/** \class RawToAttenuationImageFilter
 * \brief Convert raw XRad data to attenuation images
 *
 * \author Simon Rit
 *
 * \ingroup RTK ImageToImageFilter
 */
namespace rtk
{

template <class TInputImage, class TOutputImage = TInputImage>
class ITK_EXPORT XRadRawToAttenuationImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(XRadRawToAttenuationImageFilter);

  /** Standard class type alias. */
  using Self = XRadRawToAttenuationImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using OutputImagePointer = typename OutputImageType::Pointer;
  using OutputImageRegionType = typename TOutputImage::RegionType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(XRadRawToAttenuationImageFilter, itk::ImageToImageFilter);

protected:
  XRadRawToAttenuationImageFilter();
  ~XRadRawToAttenuationImageFilter() override = default;

  void
  BeforeThreadedGenerateData() override;

  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

private:
  OutputImagePointer m_DarkImage;
  OutputImagePointer m_FlatImage;
  std::string        m_DarkImageFileName;
  std::string        m_FlatImageFileName;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkXRadRawToAttenuationImageFilter.hxx"
#endif

#endif
