#ifndef __giftDaubechiesWaveletKernelSource_H
#define __giftDaubechiesWaveletKernelSource_H

//Includes
#include<itkImageToImageFilter.h>

namespace gift {

/**
 * \class Daubechies Wavelet Kernel Source
 * \brief Creates a Daubechies wavelets kernel image with the requested
 * attributes (order, type, pass along each dimension)
 *
 */
template<TImage>
class DaubechiesWaveletsKernelSource
    : public itk::ImageToImageFilter<TImage>
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
    typedef DaubechiesWaveletKernelSource Self;
    typedef itk::ImageToImageFilter<TImage>  Superclass;

    /** Default constructor (order = 3) */
    DaubechiesWaveletKernelSource()
    {
        //Set defaults
        this->SetLowpassDeconstruction();
        this->m_Order = 3;
    }

    /** Constructor */
    DaubechiesWaveletKernelSource(unsigned int order)
    {
        //Set defaults
        this->SetDeconstruction();
        this->m_Order = order;
    }

    /** Sets the filter to return coefficients for low pass, deconstruct. */
    void SetDeconstruction()
    {
        m_Type = Self::Deconstruct;
    }

    /** Sets the filter to return coefficients for low pass, reconstruct. */
    void SetReconstruction()
    {
        m_Type = Self::Reconstruct;
    }

    /** Prints some debugging information. */
    virtual void PrintSelf(std::ostream& os, itk::Indent i)
    {
        os  << i << "DaubechiesWaveletKernelSource { this=" << this
            << " }" << std::endl;

        os << i << "m_Order=" << this->GetOrder() << std::endl;
        os << i << "m_Pass=" << std::endl;
        for (unsigned int dim=0; dim<TImage::ImageDimension; dim++)
          {
          os << i << i << this->m_Pass[dim] << std::endl;
          }
        os << i << "m_Type=" << this->m_Type << std::endl;

        Superclass::PrintSelf( os, i.GetNextIndent() );
    }

protected:
    typedef typename Superclass::CoefficientVector CoefficientVector;

    /** Calculates CoefficientsVector coefficients. */
    CoefficientVector GenerateCoefficients();

    /** Does the real work */
    void GenerateData();

private:

    /** Returns the wavelet coefficients for each type*/
    CoefficientVector GenerateCoefficientsLowpassDeconstruct();
    CoefficientVector GenerateCoefficientsHighpassDeconstruct();
    CoefficientVector GenerateCoefficientsLowpassReconstruct();
    CoefficientVector GenerateCoefficientsHighpassReconstruct();

    /** Specifies the wavelet type name */
    unsigned int m_Order;

    /** Specifies the filter pass along each dimension */
    Pass *m_Pass;

    /** Specifies the filter type */
    Type m_Type;
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftDaubechiesWaveletKernelSource.txx"
#endif

#endif
