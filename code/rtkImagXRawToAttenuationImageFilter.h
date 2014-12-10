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

#ifndef __rtkImagXRawToAttenuationImageFilter_h
#define __rtkImagXRawToAttenuationImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkCropImageFilter.h>

#include "rtkBoellaardScatterCorrectionImageFilter.h"
#include "rtkLUTbasedVariableI0RawToAttenuationImageFilter.h"

namespace rtk
{

/** \class ImagXRawToAttenuationImageFilter
 * \brief Convert raw ImagX data to attenuation images
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */

template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT ImagXRawToAttenuationImageFilter :
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ImagXRawToAttenuationImageFilter                   Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ImagXRawToAttenuationImageFilter, itk::ImageToImageFilter);
protected:
  ImagXRawToAttenuationImageFilter();
  ~ImagXRawToAttenuationImageFilter(){
  }

  /** Apply changes to the input image requested region. */
  virtual void GenerateInputRequestedRegion();

  void GenerateOutputInformation();

  /** Single-threaded version of GenerateData.  This filter delegates
   * to other filters. */
  void GenerateData();

private:
  //purposely not implemented
  ImagXRawToAttenuationImageFilter(const Self&);
  void operator=(const Self&);

  typedef itk::CropImageFilter<InputImageType, InputImageType>                       CropFilterType;
  typedef rtk::BoellaardScatterCorrectionImageFilter<InputImageType, InputImageType> ScatterFilterType;
  typedef rtk::LUTbasedVariableI0RawToAttenuationImageFilter<InputImageType,
                                                             OutputImageType>        LookupTableFilterType;

  typename LookupTableFilterType::Pointer m_LookupTableFilter;
  typename CropFilterType::Pointer        m_CropFilter;
  typename ScatterFilterType::Pointer     m_ScatterFilter;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkImagXRawToAttenuationImageFilter.txx"
#endif

#endif
