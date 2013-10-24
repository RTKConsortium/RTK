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

#ifndef __rtkElektaSynergyRawToAttenuationImageFilter_h
#define __rtkElektaSynergyRawToAttenuationImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkCropImageFilter.h>

#include "rtkElektaSynergyLookupTableImageFilter.h"
#include "rtkBoellaardScatterCorrectionImageFilter.h"

namespace rtk
{

/** \class ElektaSynergyRawToAttenuationImageFilter
 * \brief Convert raw Elekta Synergy data to attenuation images
 *
 * This composite filter composes the operations required to convert
 * a raw image from the Elekta Synergy cone-beam CT scanner to
 * attenuation images usable in standard reconstruction algorithms,*
 * e.g. Feldkamp algorithm.
 *
 * \test rtkelektatest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT ElektaSynergyRawToAttenuationImageFilter :
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ElektaSynergyRawToAttenuationImageFilter           Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ElektaSynergyRawToAttenuationImageFilter, itk::ImageToImageFilter);

protected:
  ElektaSynergyRawToAttenuationImageFilter();
  ~ElektaSynergyRawToAttenuationImageFilter(){
  }

  /** Apply changes to the input image requested region. */
  virtual void GenerateInputRequestedRegion();

  void GenerateOutputInformation();

  /** Single-threaded version of GenerateData.  This filter delegates
   * to other filters. */
  void GenerateData();

private:
  //purposely not implemented
  ElektaSynergyRawToAttenuationImageFilter(const Self&);
  void operator=(const Self&);

  typedef itk::CropImageFilter<InputImageType, InputImageType>                                  CropFilterType;
  typedef rtk::BoellaardScatterCorrectionImageFilter<InputImageType, InputImageType>            ScatterFilterType;
  typedef typename rtk::ElektaSynergyRawLookupTableImageFilter<OutputImageType::ImageDimension> RawLookupTableFilterType;
  typedef rtk::ElektaSynergyLogLookupTableImageFilter<OutputImageType>                          LogLookupTableFilterType;

  typename RawLookupTableFilterType::Pointer m_RawLookupTableFilter;
  typename LogLookupTableFilterType::Pointer m_LogLookupTableFilter;
  typename CropFilterType::Pointer           m_CropFilter;
  typename ScatterFilterType::Pointer        m_ScatterFilter;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkElektaSynergyRawToAttenuationImageFilter.txx"
#endif

#endif
