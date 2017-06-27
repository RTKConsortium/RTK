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
 * \ingroup ImageToImageFilter
 */
namespace rtk
{

template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT XRadRawToAttenuationImageFilter :
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef XRadRawToAttenuationImageFilter                    Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                       InputImageType;
  typedef TOutputImage                      OutputImageType;
  typedef typename OutputImageType::Pointer OutputImagePointer;
  typedef typename TOutputImage::RegionType OutputImageRegionType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(XRadRawToAttenuationImageFilter, itk::ImageToImageFilter);

protected:
  XRadRawToAttenuationImageFilter();
  ~XRadRawToAttenuationImageFilter() {}

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

private:
  //purposely not implemented
  XRadRawToAttenuationImageFilter(const Self&);
  void operator=(const Self&);

  OutputImagePointer m_DarkImage;
  OutputImagePointer m_FlatImage;
  std::string        m_DarkImageFileName;
  std::string        m_FlatImageFileName;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkXRadRawToAttenuationImageFilter.hxx"
#endif

#endif
