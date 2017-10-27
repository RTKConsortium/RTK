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

namespace rtk
{

/** \class EdfRawToAttenuationImageFilter
 * \brief Convert raw ESRF data to attenuation images
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */

template<class TInputImage, class TOutputImage=TInputImage>
#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR > 4)
class ITKIOImageBase_HIDDEN EdfRawToAttenuationImageFilter :
#else
class ITK_EXPORT EdfRawToAttenuationImageFilter :
#endif
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef EdfRawToAttenuationImageFilter                     Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                       InputImageType;
  typedef TOutputImage                      OutputImageType;
  typedef typename TOutputImage::RegionType OutputImageRegionType;

  typedef  std::vector<std::string> FileNamesContainer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(EdfRawToAttenuationImageFilter, itk::ImageToImageFilter);

  /** Set the vector of strings that contains the file names. Files
   * are processed in sequential order. */
  void SetFileNames(const FileNamesContainer &name)
  {
    if ( m_FileNames != name)
      {
      m_FileNames = name;
      this->Modified();
      }
  }

  const FileNamesContainer & GetFileNames() const
  {
    return m_FileNames;
  }

protected:
  EdfRawToAttenuationImageFilter();
  ~EdfRawToAttenuationImageFilter() {}

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

private:
  //purposely not implemented
  EdfRawToAttenuationImageFilter(const Self&);
  void operator=(const Self&);

  typedef itk::ImageSeriesReader< InputImageType > EdfImageSeries;
  typename EdfImageSeries::Pointer m_DarkProjectionsReader;
  typename EdfImageSeries::Pointer m_ReferenceProjectionsReader;

  /** A list of filenames from which the input was read. */
  FileNamesContainer m_FileNames;

  /** The list of indices of the input for each reference image. */
  std::vector<typename InputImageType::IndexValueType> m_ReferenceIndices;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkEdfRawToAttenuationImageFilter.hxx"
#endif

#endif
