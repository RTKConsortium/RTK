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

#ifndef rtkEdfRawToAttenuationImageFilter_h
#define rtkEdfRawToAttenuationImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkImageSeriesReader.h>

#include "rtkConfiguration.h"

namespace rtk
{

/** \class EdfRawToAttenuationImageFilter
 * \brief Convert raw ESRF data to attenuation images
 *
 * \author Simon Rit
 *
 * \ingroup RTK ImageToImageFilter
 */

template <class TInputImage, class TOutputImage = TInputImage>
class ITKIOImageBase_HIDDEN EdfRawToAttenuationImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(EdfRawToAttenuationImageFilter);

  /** Standard class type alias. */
  using Self = EdfRawToAttenuationImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using OutputImageRegionType = typename TOutputImage::RegionType;

  using FileNamesContainer = std::vector<std::string>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(EdfRawToAttenuationImageFilter, itk::ImageToImageFilter);

  /** Set the vector of strings that contains the file names. Files
   * are processed in sequential order. */
  void
  SetFileNames(const FileNamesContainer & name)
  {
    if (m_FileNames != name)
    {
      m_FileNames = name;
      this->Modified();
    }
  }

  const FileNamesContainer &
  GetFileNames() const
  {
    return m_FileNames;
  }

protected:
  EdfRawToAttenuationImageFilter();
  ~EdfRawToAttenuationImageFilter() override = default;

  void
  BeforeThreadedGenerateData() override;

  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

private:
  using EdfImageSeries = itk::ImageSeriesReader<InputImageType>;
  typename EdfImageSeries::Pointer m_DarkProjectionsReader;
  typename EdfImageSeries::Pointer m_ReferenceProjectionsReader;

  /** A list of filenames from which the input was read. */
  FileNamesContainer m_FileNames;

  /** The list of indices of the input for each reference image. */
  std::vector<typename InputImageType::IndexValueType> m_ReferenceIndices;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkEdfRawToAttenuationImageFilter.hxx"
#endif

#endif
