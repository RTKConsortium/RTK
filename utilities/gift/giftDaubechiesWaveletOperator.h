#ifndef __giftDaubechiesWaveletOperator_H
#define __giftDaubechiesWaveletOperator_H

//Includes
#include "itkNeighborhoodOperator.h"
#include <math.h>

namespace gift {

/**
 * \class Daubechies Wavelet Operator
 * \brief A NeighborhoodOperator whose coefficients are a one
 * dimensional, daubechies wavelet kernels.
 *
 * \ingroup Operators
 */
template<class TPixel, unsigned int VDimension=2,
    class TAllocator = itk::NeighborhoodAllocator<TPixel> >
class DaubechiesWaveletOperator
    : public itk::NeighborhoodOperator<TPixel, VDimension, TAllocator>
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
    typedef DaubechiesWaveletOperator Self;
    typedef itk::NeighborhoodOperator<TPixel, VDimension, TAllocator>  Superclass;

    /** Default constructor (order = 3) */
    DaubechiesWaveletOperator()
    {
        //Set defaults
        this->SetLowpassDeconstruction();
        this->m_Order = 3;
    }

    /** Constructor */
    DaubechiesWaveletOperator(unsigned int order)
    {
        //Set defaults
        this->SetLowpassDeconstruction();
        this->m_Order = order;
    }

    /** Sets the filter to return coefficients for low pass, deconstruct. */
    void SetLowpassDeconstruction()
    {
        m_Pass = Self::Low;
        m_Type = Self::Deconstruct;
    }

    /** Sets the filter to return coefficients for high pass, deconstruct. */
    void SetHighpassDeconstruction()
    {
        m_Pass = Self::High;
        m_Type = Self::Deconstruct;
    }

    /** Sets the filter to return coefficients for low pass, reconstruct. */
    void SetLowpassReconstruction()
    {
        m_Pass = Self::Low;
        m_Type = Self::Reconstruct;
    }

    /** Sets the filter to return coefficients for high pass, reconstruct. */
    void SetHighpassReconstruction()
    {
        m_Pass = Self::High;
        m_Type = Self::Reconstruct;
    }

    /** Gets the name of the bior wavelet */
    unsigned int GetOrder()
    {
        return this->m_Order;
    }


    /** Prints some debugging information. */
    virtual void PrintSelf(std::ostream& os, itk::Indent i)
    {
        os  << i << "DaubechiesWaveletOperator { this=" << this
            << " }" << std::endl;

        os << i << "m_Order=" << this->GetOrder() << std::endl;
        os << i << "m_Pass=" << this->m_Pass << std::endl;
        os << i << "m_Type=" << this->m_Type << std::endl;

        Superclass::PrintSelf( os, i.GetNextIndent() );
    }

protected:
    typedef typename Superclass::CoefficientVector CoefficientVector;

    /** Calculates operator coefficients. */
    CoefficientVector GenerateCoefficients();

    /** Arranges coefficients spatially in the memory buffer. */
    void Fill(const CoefficientVector& coeff)
    {    this->FillCenteredDirectional(coeff);  }

private:

    /** Returns the wavelet coefficients for each type*/
    CoefficientVector GenerateCoefficientsLowpassDeconstruct();
    CoefficientVector GenerateCoefficientsHighpassDeconstruct();
    CoefficientVector GenerateCoefficientsLowpassReconstruct();
    CoefficientVector GenerateCoefficientsHighpassReconstruct();

    /** Specifies the wavelet type name */
    unsigned int m_Order;

    /** Specifies the filter pass */
    Pass m_Pass;

    /** Specifies the filter type */
    Type m_Type;
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftDaubechiesWaveletOperator.txx"
#endif

#endif
