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
#include "rtkWaterPrecorrectionImageFilter.h"

#include <vector>

using namespace std;

namespace rtk
{

  /** \class ImagXRawToAttenuationImageFilter
  * \brief Convert raw ImagX data to attenuation images
  *
  * \author Simon Rit, S. Brousmiche
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
    typedef typename InputImageType::SizeType                                  InputImageSizeType;
    typedef TOutputImage                                                       OutputImageType;

    typedef std::vector< unsigned int >       BinParamType;
    typedef std::vector< double >             WpcVectorType;
        
    /** Standard New method. */
    itkNewMacro(Self);

    /** Runtime information support. */
    itkTypeMacro(ImagXRawToAttenuationImageFilter, itk::ImageToImageFilter);

    /** Set the cropping sizes for the upper and lower boundaries. */
    itkSetMacro(UpperBoundaryCropSize, InputImageSizeType);
    itkGetMacro(UpperBoundaryCropSize, InputImageSizeType);
    itkSetMacro(LowerBoundaryCropSize, InputImageSizeType);
    itkGetMacro(LowerBoundaryCropSize, InputImageSizeType);

    /** Set the binning kernel size. */
    itkGetMacro(BinningKernelSize, BinParamType);
    virtual void SetBinningKernelSize(const std::vector<unsigned int> binfactor) 
    {
      if (this->m_BinningKernelSize != binfactor)
      {
        this->m_BinningKernelSize = binfactor;
        this->Modified();
      }
    }

    /** Get / Set the I0 estimation parameters. */
    itkSetMacro(ExpectedI0, unsigned short);
    itkGetMacro(ExpectedI0, unsigned short);
    itkSetMacro(ForceExpectedValue, bool);
    itkGetConstMacro(ForceExpectedValue, bool);
    itkBooleanMacro(ForceExpectedValue);

    /** Set the scatter correction parameters. */
    itkGetMacro(RelativeAirThreshold, double);
    itkGetMacro(ScatterToPrimaryRatio, double);
    void SetScatterCorrectionParameters(const double relativeAirThreshold, const double scatterToPrimaryRatio)
    {
      if ((this->m_RelativeAirThreshold != relativeAirThreshold) && (this->m_ScatterToPrimaryRatio != scatterToPrimaryRatio))
      {
        this->m_RelativeAirThreshold = relativeAirThreshold;
        this->m_ScatterToPrimaryRatio = scatterToPrimaryRatio;
        this->Modified();
      }
    }

    /** Get / Set the water precorrection parameters. */
    itkGetMacro(WpcCoefficients, WpcVectorType);
    virtual void SetWpcCoefficients(const WpcVectorType _arg)
    {
      if (this->m_WpcCoefficients != _arg)
      {
        this->m_WpcCoefficients = _arg;
        this->Modified();
      }
    }
    
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

    typedef itk::ExtractImageFilter<InputImageType, InputImageType>                     ExtractFilterType;
    typedef itk::CropImageFilter<InputImageType, InputImageType>                        CropFilterType;
    typedef itk::BinShrinkImageFilter<InputImageType, InputImageType>                   BinningFilterType;
    typedef rtk::BoellaardScatterCorrectionImageFilter<InputImageType, InputImageType>  ScatterFilterType;
    typedef rtk::I0EstimationProjectionFilter<InputImageType, InputImageType, bitShift> I0FilterType;
    typedef rtk::LUTbasedVariableI0RawToAttenuationImageFilter<InputImageType,
                                                               OutputImageType>        LookupTableFilterType;
    typedef rtk::WaterPrecorrectionImageFilter<OutputImageType, OutputImageType>       WpcType;
    typedef rtk::ConstantImageSource<OutputImageType>                                  ConstantImageSourceType;
    typedef itk::PasteImageFilter<OutputImageType, OutputImageType>                    PasteFilterType;

    typename ExtractFilterType::Pointer       m_ExtractFilter;
    typename CropFilterType::Pointer          m_CropFilter;
    typename BinningFilterType::Pointer       m_BinningFilter;
    typename ScatterFilterType::Pointer       m_ScatterFilter;
    typename I0FilterType::Pointer            m_I0estimationFilter;
    typename LookupTableFilterType::Pointer   m_LookupTableFilter;
    typename WpcType::Pointer                 m_WpcFilter;
    typename PasteFilterType::Pointer         m_PasteFilter;
    typename ConstantImageSourceType::Pointer m_ConstantSource;

    /** Extraction regions for extract filter */
    typename InputImageType::RegionType m_ExtractRegion;
    typename InputImageType::RegionType m_PasteRegion;

    /** Crop filter parameters */
    InputImageSizeType m_LowerBoundaryCropSize;
    InputImageSizeType m_UpperBoundaryCropSize;

    /** Binning filter parameters */
    BinParamType m_BinningKernelSize;

    /** I0 estimation parameters */
    unsigned short m_ExpectedI0;
    bool m_ForceExpectedValue;       // No reestimation of I0

    /** Scatter correction parameters */
    double m_RelativeAirThreshold;
    double m_ScatterToPrimaryRatio;

    /** Water pre-correction parameters */
    WpcVectorType m_WpcCoefficients;
    
  }; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkImagXRawToAttenuationImageFilter.txx"
#endif

#endif
