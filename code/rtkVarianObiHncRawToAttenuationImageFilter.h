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

#ifndef __rtkVarianObiHncRawToAttenuationImageFilter_h
#define __rtkVarianObiHncRawToAttenuationImageFilter_h

#include <itkImageToImageFilter.h>
 
namespace rtk
{

/** \class VarianObiHncRawToAttenuationImageFilter
 * \brief Converts a raw value from the HNC format measured by the Varian OBI system to attenuation
 *
 * The current implementation assues a norm chamber factor of 300. Because both the flood field
 * and the projection therefore have the same norm chamber factor, this is unused.
 *
 * Potentially, because HNC PixelType is unsigned short, this class could be converted to the LUT-based
 * raw to attenuation filter similar to Elekta HIS.
 *
 * \author Geoff Hugo, VCU
 *
 * \ingroup ImageToImageFilter
 */

template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT VarianObiHncRawToAttenuationImageFilter :
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef VarianObiHncRawToAttenuationImageFilter                     Self;
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
  itkTypeMacro(VarianObiHncRawToAttenuationImageFilter, itk::ImageToImageFilter);

  /** Get/Set the filename for path. */
  itkSetMacro(ProjectionFileName, std::string);
  itkGetMacro(ProjectionFileName, std::string);

protected:
  VarianObiHncRawToAttenuationImageFilter();
  ~VarianObiHncRawToAttenuationImageFilter(){
  }

  void BeforeThreadedGenerateData();

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

private:
  //purposely not implemented
  VarianObiHncRawToAttenuationImageFilter(const Self&);
  void operator=(const Self&);

  std::string m_FloodImageFileName;
  std::string m_ProjectionFileName;
  typename InputImageType::Pointer m_FloodImage;

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkVarianObiHncRawToAttenuationImageFilter.hxx"
#endif

#endif
