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

#ifndef rtkDaubechiesWaveletsConvolutionImageFilter_h
#define rtkDaubechiesWaveletsConvolutionImageFilter_h

//Includes
#include <itkImageToImageFilter.h>
#include <itkConvolutionImageFilter.h>

#include "rtkMacro.h"

namespace rtk {

/**
 * \class DaubechiesWaveletsConvolutionImageFilter
 * \brief Creates a Daubechies wavelets kernel image with the requested
 * attributes (order, type, pass along each dimension)
 *
 * This filter is inspired from Dan Mueller's GIFT package
 * http://www.insight-journal.org/browse/publication/103
 *
 * \author Cyril Mory
 *
 */
template<typename TImage>
class DaubechiesWaveletsConvolutionImageFilter : public itk::ImageToImageFilter<TImage, TImage>
{
public:

    enum Pass
    {
        Low = 0x0,  //Indicates to return the low-pass filter coefficients
        High= 0x1   //Indicates to return the high-pass filter coefficients
    };

    enum Type
    {
        Deconstruct = 0x0,  //Indicates to deconstruct the image into levels/bands
        Reconstruct = 0x1   //Indicates to reconstruct the image from levels/bands
    };


    /** Standard class typedefs. */
    typedef DaubechiesWaveletsConvolutionImageFilter  Self;
    typedef itk::ImageToImageFilter<TImage, TImage>   Superclass;
    typedef itk::SmartPointer<Self>                   Pointer;
    typedef itk::SmartPointer<const Self>             ConstPointer;

    /** Typedef for the output image type. */
    typedef TImage OutputImageType;

    /** Typedef for the output image PixelType. */
    typedef typename TImage::PixelType OutputImagePixelType;

    /** Typedef to describe the output image region type. */
    typedef typename TImage::RegionType OutputImageRegionType;

    /** Typedef for the "pass" vector (high pass or low pass along each dimension). */
    typedef typename itk::Vector<typename Self::Pass, TImage::ImageDimension> PassVector;

    /** Run-time type information (and related methods). */
    itkTypeMacro(DaubechiesWaveletsConvolutionImageFilter, itk::ImageSource)

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Typedef for the internal convolution filter */
    typedef typename itk::ConvolutionImageFilter<TImage> ConvolutionFilterType;

    /** Sets the filter to return coefficients for low pass, deconstruct. */
    void SetDeconstruction();

    /** Sets the filter to return coefficients for low pass, reconstruct. */
    void SetReconstruction();

    /** Prints some debugging information. */
    virtual void PrintSelf(std::ostream& os, itk::Indent i);

    /** Set and Get macro for the wavelet order */
    itkSetMacro(Order, unsigned int)
    itkGetMacro(Order, unsigned int)

    /** Set and Get macro for the pass vector */
    itkSetMacro(Pass, PassVector)
    itkGetMacro(Pass, PassVector)

protected:
    DaubechiesWaveletsConvolutionImageFilter();
    ~DaubechiesWaveletsConvolutionImageFilter();

    typedef std::vector<typename TImage::PixelType> CoefficientVector;

    /** Calculates CoefficientsVector coefficients. */
    CoefficientVector GenerateCoefficients();

    /** Does the real work */
    void GenerateData() ITK_OVERRIDE;

    /** Defines the size, spacing, ... of the output kernel image */
    void GenerateOutputInformation() ITK_OVERRIDE;

private:

    /** Returns the wavelet coefficients for each type*/
    CoefficientVector GenerateCoefficientsLowpassDeconstruct();
    CoefficientVector GenerateCoefficientsHighpassDeconstruct();
    CoefficientVector GenerateCoefficientsLowpassReconstruct();
    CoefficientVector GenerateCoefficientsHighpassReconstruct();

    /** Specifies the wavelet type name */
    unsigned int m_Order;

    /** Specifies the filter pass along each dimension */
    PassVector m_Pass;

    /** Specifies the filter type */
    Type m_Type;
};

}// namespace rtk

//Include CXX
#ifndef rtk_MANUAL_INSTANTIATION
#include "rtkDaubechiesWaveletsConvolutionImageFilter.hxx"
#endif

#endif
