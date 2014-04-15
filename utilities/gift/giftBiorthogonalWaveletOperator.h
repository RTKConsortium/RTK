/*=========================================================================

  Program:  GIFT Biorthogonal Wavelet Operator
  Module:   giftBiorthogonalWaveletOperator.h
  Language: C++
  Date:     2005/10/21
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftBiorthogonalWaveletOperator_H
#define __giftBiorthogonalWaveletOperator_H

//Includes
#include "itkNeighborhoodOperator.h"
#include <math.h>

namespace gift {

/**
 * \class Biorthogonal Wavelet Operator
 * \brief A NeighborhoodOperator whose coefficients are a one
 * dimensional, biorthogonal wavelet kernels.
 *
 * \ingroup Operators
 */
template<class TPixel, unsigned int VDimension=2,
    class TAllocator = itk::NeighborhoodAllocator<TPixel> >
class BiorthogonalWaveletOperator
    : public itk::NeighborhoodOperator<TPixel, VDimension, TAllocator>
{
public:
    
    /** 
    * Enumerates the Biorthogonal Wavelet Names
    * (pattern = [family_major_minor]).
    *
    * NOTE: The assignment of the enum value is such that the first byte
    *       contains the major identifier, and the second by the minor 
    *       identifier.
    */
    enum Name 
    { 
        Bior_1_1 = 0x11,
        Bior_1_3 = 0x13,
        Bior_1_5 = 0x15,
        Bior_2_2 = 0x22,
        Bior_2_4 = 0x24,
        Bior_2_6 = 0x26, //Not yet implemented
        Bior_2_8 = 0x28, //Not yet implemented
        Bior_3_1 = 0x31,
        Bior_3_3 = 0x33,
        Bior_3_5 = 0x35,
        Bior_3_7 = 0x37, //Not yet implemented
        Bior_3_9 = 0x39, //Not yet implemented
        Bior_4_4 = 0x44, //Not yet implemented
        Bior_5_5 = 0x55, //Not yet implemented
        Bior_6_8 = 0x68  //Not yet implemented
    };
  
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
    typedef BiorthogonalWaveletOperator Self;
    typedef itk::NeighborhoodOperator<TPixel, VDimension, TAllocator>  Superclass;
            
    /** Default constructor (Bior_2_2) */
    BiorthogonalWaveletOperator()
    { 
        //Set defaults
        this->SetLowpassDeconstruction();
        this->m_Name = gift::BiorthogonalWaveletOperator<TPixel, VDimension>::Bior_2_2;
    }

    
    /** Constructor */
    BiorthogonalWaveletOperator(Name name)
    { 
        //Set defaults
        this->SetLowpassDeconstruction();
        this->m_Name = name;
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
    Name GetName()
    {
        return this->m_Name;
    }
    
    /** Gets the major identifier given by a bior wavelet enum */
    unsigned int GetNameMajor();

    /** Gets the minor identifier given by a bior wavelet enum */
    unsigned int GetNameMinor();

    /** Prints some debugging information. */
    virtual void PrintSelf(std::ostream& os, itk::Indent i)
    {
        os  << i << "BiorthogonalWaveletOperator { this=" << this
            << " }" << std::endl;

        os << i << "m_Name=Bior " << this->GetNameMajor() << "." << this->GetNameMinor() << std::endl;
        os << i << "m_Pass=" << this->m_Pass << std::endl;
        os << i << "m_Type=" << this->m_Type << std::endl;

        //CoefficientVector coefficients = GenerateCoefficients();
        //os << i << "m_Coefficients: [ ";
        //for (CoefficientVector::iterator it=coefficients.begin(); it<coefficients.end(); ++it) os << *it << " ";
        //os << "]" << std::endl;
        Superclass::PrintSelf( os, i.GetNextIndent() );
    }
  
protected:
    typedef typename Superclass::CoefficientVector CoefficientVector;

    /** Calculates operator coefficients. */
    CoefficientVector GenerateCoefficients();

    /** Arranges coefficients spatially in the memory buffer. */
    void Fill(const CoefficientVector& coeff)
    {    this->FillCenteredDirectional(coeff);  }

    /** Exposes a bit mask for the major id of the name */
    const static unsigned int NAME_MASK_MAJOR = 0xF0; //1111 0000

    /** Exposes a bit mask for the minor id of the name */
    const static unsigned int NAME_MASK_MINOR = 0x0F; //0000 1111

private:

    /** Returns the wavelet coefficients for each type*/
    CoefficientVector GenerateCoefficientsLowpassDeconstruct();
    CoefficientVector GenerateCoefficientsHighpassDeconstruct();
    CoefficientVector GenerateCoefficientsLowpassReconstruct();
    CoefficientVector GenerateCoefficientsHighpassReconstruct();

    /** Specifies the wavelet type name */
    Name m_Name;

    /** Specifies the filter pass */
    Pass m_Pass;
      
    /** Specifies the filter type */
    Type m_Type;
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftBiorthogonalWaveletOperator.txx"
#endif

#endif
