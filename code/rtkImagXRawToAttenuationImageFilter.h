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
#include <itkBinShrinkImageFilter.h>
#include <itkCropImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>
#include <rtkConstantImageSource.h>

#include "rtkBoellaardScatterCorrectionImageFilter.h"
#include "rtkI0EstimationProjectionFilter.h"
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

  template<class TOutputImage, unsigned char bitShift = 2>
  class ITK_EXPORT ImagXRawToAttenuationImageFilter :
    public itk::ImageToImageFilter < typename itk::Image< unsigned short, TOutputImage::ImageDimension >, TOutputImage >
  {
  public:
    /** Standard class typedefs. */
    typedef ImagXRawToAttenuationImageFilter                   Self;
    typedef itk::ImageToImageFilter<itk::Image<unsigned short, TOutputImage::ImageDimension>,
                                    TOutputImage>              Superclass;
    typedef itk::SmartPointer<Self>                            Pointer;
    typedef itk::SmartPointer<const Self>                      ConstPointer;

    /** Some convenient typedefs. */
    typedef typename itk::Image<unsigned short, TOutputImage::ImageDimension>  InputImageType;
    typedef TOutputImage                                       OutputImageType;

    /** Standard New method. */
    itkNewMacro(Self);

    /** Runtime information support. */
    itkTypeMacro(ImagXRawToAttenuationImageFilter, itk::ImageToImageFilter);
  protected:
    ImagXRawToAttenuationImageFilter();
    ~ImagXRawToAttenuationImageFilter(){
    }

    void GenerateOutputInformation();

    /** Single-threaded version of GenerateData.  This filter delegates
    * to other filters. */
    void GenerateData();

  private:
    //purposely not implemented
    ImagXRawToAttenuationImageFilter(const Self&);
    void operator=(const Self&);

    typedef itk::ExtractImageFilter<InputImageType, InputImageType>                    ExtractFilterType;
    typedef itk::CropImageFilter<InputImageType, InputImageType>                       CropFilterType;
    typedef itk::BinShrinkImageFilter<InputImageType, InputImageType>                  BinningFilterType;
    typedef rtk::BoellaardScatterCorrectionImageFilter<InputImageType, InputImageType> ScatterFilterType;
    typedef rtk::I0EstimationProjectionFilter<bitShift>                                I0FilterType;
    typedef rtk::LUTbasedVariableI0RawToAttenuationImageFilter<InputImageType,
                                                               OutputImageType>        LookupTableFilterType;
    typedef rtk::ConstantImageSource<OutputImageType>                                  ConstantImageSourceType;
    typedef itk::PasteImageFilter<OutputImageType, OutputImageType>                    PasteFilterType;

    typename ExtractFilterType::Pointer     m_ExtractFilter;
    typename CropFilterType::Pointer        m_CropFilter;
    typename BinningFilterType::Pointer     m_BinningFilter;
    typename ScatterFilterType::Pointer     m_ScatterFilter;
    typename I0FilterType::Pointer          m_I0estimationFilter;
    typename LookupTableFilterType::Pointer m_LookupTableFilter;
    typename PasteFilterType::Pointer       m_PasteFilter;
    typename ConstantImageSourceType::Pointer m_ConstantSource;

    /** Extraction regions for extract filter */
    typename InputImageType::RegionType m_ExtractRegion;
    typename InputImageType::RegionType m_PasteRegion;

  }; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkImagXRawToAttenuationImageFilter.txx"
#endif

#endif
