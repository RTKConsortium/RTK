#ifndef __giftDaubechiesWaveletKernelSource_H
#define __giftDaubechiesWaveletKernelSource_H

//Includes
#include<itkImageSource.h>

namespace gift {

/**
 * \class Daubechies Wavelet Kernel Source
 * \brief Creates a Daubechies wavelets kernel image with the requested
 * attributes (order, type, pass along each dimension)
 *
 */
template<TImage>
class DaubechiesWaveletsKernelSource
    : public itk::ImageSource<TImage>
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
    typedef itk::ImageSource<TImage>  Superclass;


    /** Sets the filter to return coefficients for low pass, deconstruct. */
    void SetDeconstruction();

    /** Sets the filter to return coefficients for low pass, reconstruct. */
    void SetReconstruction();

    /** Prints some debugging information. */
    virtual void PrintSelf(std::ostream& os, itk::Indent i);

    /** Set and Get macro for the pass vector */
    itkSetMacro(Pass, Pass*)
    itkGetMacro(Pass, Pass*)

protected:
    typedef std::vector<typename TImage::PixelType> CoefficientVector;

    /** Calculates CoefficientsVector coefficients. */
    CoefficientVector GenerateCoefficients();

    /** Does the real work */
    virtual void GenerateData();

    /** Defines the size, spacing, ... of the output kernel image */
    virtual void GenerateOutputInformation();

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
#include "giftDaubechiesWaveletsKernelSource.txx"
#endif

#endif
