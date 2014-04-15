/*=========================================================================

  Program:  GIFT Biorthogonal Wavelet Operator
  Module:   giftBiorthogonalWaveletOperator.txx
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
#ifndef _giftBiorthogonalWaveletOperator_TXX
#define _giftBiorthogonalWaveletOperator_TXX

//Includes
#include "giftBiorthogonalWaveletOperator.h"
#include "vnl/vnl_math.h"
#include <vector>
#include <algorithm>

namespace gift
{

template<class TPixel, unsigned int VDimension, class TAllocator>
typename BiorthogonalWaveletOperator<TPixel, VDimension, TAllocator>
::CoefficientVector
BiorthogonalWaveletOperator<TPixel, VDimension, TAllocator>
::GenerateCoefficients()
{
    //Test type
    switch (this->m_Pass)
    {
    case Low:
        switch (this->m_Type)
        {
        case Deconstruct:     //Low-pass, Deconstruct
            return this->GenerateCoefficientsLowpassDeconstruct();
            break;
        case Reconstruct:     //Low-pass, Reconstruct
            return this->GenerateCoefficientsLowpassReconstruct();
            break;
        }
        break;
    case High:
        switch (this->m_Type)
        {
        case Deconstruct:     //High-pass, Deconstruct
            return this->GenerateCoefficientsHighpassDeconstruct();
            break;
        case Reconstruct:     //High-pass, Reconstruct
            return this->GenerateCoefficientsHighpassReconstruct();
            break;
        }
        break;
    }

    //Default return
    CoefficientVector coeff;
    return coeff;
}

template<class TPixel, unsigned int VDimension, class TAllocator>
unsigned int
BiorthogonalWaveletOperator<TPixel, VDimension, TAllocator>
::GetNameMajor()
{
    unsigned int temp = (unsigned int)(BiorthogonalWaveletOperator::NAME_MASK_MAJOR & this->m_Name);
    temp = temp >> 4;
    return temp;
}

template<class TPixel, unsigned int VDimension, class TAllocator>
unsigned int
BiorthogonalWaveletOperator<TPixel, VDimension, TAllocator>
::GetNameMinor()
{
    return (unsigned int)(BiorthogonalWaveletOperator::NAME_MASK_MINOR & this->m_Name);
}

template<class TPixel, unsigned int VDimension, class TAllocator>
typename BiorthogonalWaveletOperator<TPixel, VDimension, TAllocator>::
CoefficientVector 
BiorthogonalWaveletOperator<TPixel, VDimension, TAllocator>
::GenerateCoefficientsLowpassDeconstruct()
{
    CoefficientVector coeff;
    
    switch (this->GetNameMajor())
    {
    //-----------------------------------------
    case 1:
        switch (this->GetNameMinor())
        {
        case 1:                             //Name_Bior_1_1
			coeff.push_back(1.0/vnl_math::sqrt2);
            coeff.push_back(1.0/vnl_math::sqrt2);
            break;
        case 3:                             //name_Bior_1_3
            coeff.push_back((-1.0/16.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/16.0)*vnl_math::sqrt2);
            coeff.push_back(1.0/vnl_math::sqrt2);
            coeff.push_back(1.0/vnl_math::sqrt2);
            coeff.push_back((1.0/16.0)*vnl_math::sqrt2);
            coeff.push_back((-1.0/16.0)*vnl_math::sqrt2);
            break;
        case 5:                             //name_Bior_1_5
            coeff.push_back((3.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((-11.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((11.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/2.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/2.0)*vnl_math::sqrt2);
            coeff.push_back((11.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((-11.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/256.0)*vnl_math::sqrt2);
            break;
        } //end case(nameMinor)
        break;
    //-----------------------------------------
    case 2:
        switch (this->GetNameMinor())
        {
        case 2:                             //Name_Bior_2_2
            coeff.push_back(0.0);
            coeff.push_back((-1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((-1.0/8.0)*vnl_math::sqrt2);
            break;
        case 4:                             //Name_Bior_2_4
            coeff.push_back(0.0);
            coeff.push_back((3.0/128.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((-1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((19.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((45.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((19.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((-1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/128.0)*vnl_math::sqrt2);
            break;
        case 6:                             //Name_Bior_2_6
            //TODO:
            std::cerr << "ERROR: Bior2.6 is NOT YET IMPLEMENTED" << std::endl;
            break;
        case 8:                             //Name_Bior_2_8
            //TODO:
            std::cerr << "ERROR: Bior2.8 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)
        break;
    //-----------------------------------------
    case 3:
        switch (this->GetNameMinor())
        {
        case 1:                             //Name_Bior_3_1
            coeff.push_back((-1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((-1.0/4.0)*vnl_math::sqrt2);
            break;
        case 3:                             //Name_Bior_3_3
            coeff.push_back((3.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((-9.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((-7.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((45.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((45.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((-7.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((-9.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/64.0)*vnl_math::sqrt2);
            break;
        case 5:                             //Name_Bior_3_5
            coeff.push_back((-5.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((15.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((19.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((-97.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((-26.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((350.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((350.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((-26.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((-97.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((19.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((15.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((-5.0/512.0)*vnl_math::sqrt2);
            break;
        case 7:                             //Name_Bior_3_7
            //TODO:
            std::cerr << "ERROR: Bior3.7 is NOT YET IMPLEMENTED" << std::endl;
            break;
        case 9:                             //Name_Bior_3_9
            //TODO:
            std::cerr << "ERROR: Bior3.9 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)         
        break;
    //-----------------------------------------
    case 4:
        switch (this->GetNameMinor())
        {
        case 4:                             //name_Bior_4_4
            //TODO:
            std::cerr << "ERROR: Bior4.4 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)
        break;
    //-----------------------------------------
    case 5:
        switch (this->GetNameMinor())
        {
        case 5:                             //Name_Bior_5_5
            //TODO:
            std::cerr << "ERROR: Bior5.5 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)         
        break;
    //-----------------------------------------
    case 6:
        switch (this->GetNameMinor())
        {
        case 8:                             //Name_Bior_6_8
            //TODO:
            std::cerr << "ERROR: Bior6.8 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)
        break;  
    } //end case(nameMajor)

    return coeff;
}

template<class TPixel, unsigned int VDimension, class TAllocator>
typename BiorthogonalWaveletOperator<TPixel, VDimension, TAllocator>::
CoefficientVector 
BiorthogonalWaveletOperator<TPixel, VDimension, TAllocator>
::GenerateCoefficientsHighpassDeconstruct()
{
    CoefficientVector coeff;
    
    switch (this->GetNameMajor())
    {
    //-----------------------------------------
    case 1:
        switch (this->GetNameMinor())
        {
        case 1:                             //Name_Bior_1_1
            coeff.push_back(-1.0/vnl_math::sqrt2);
            coeff.push_back(1.0/vnl_math::sqrt2);
            break;
        case 3:                             //name_Bior_1_3
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(-1.0/vnl_math::sqrt2);
            coeff.push_back(1.0/vnl_math::sqrt2);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            break;
        case 5:                             //name_Bior_1_5
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(-1.0/vnl_math::sqrt2);
            coeff.push_back(1.0/vnl_math::sqrt2);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            break;
        } //end case(nameMinor)
        break;
    //-----------------------------------------
    case 2:
        switch (this->GetNameMinor())
        {
        case 2:                             //Name_Bior_2_2
            coeff.push_back(0.0);
            coeff.push_back((1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((-1.0/2.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            break;
        case 4:                             //Name_Bior_2_4
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back((1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((-1.0/2.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            break;
        case 6:                             //Name_Bior_2_6
            //TODO:
            std::cerr << "ERROR: Bior2.6 is NOT YET IMPLEMENTED" << std::endl;
            break;
        case 8:                             //Name_Bior_2_8
            //TODO:
            std::cerr << "ERROR: Bior2.8 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)
        break;
    //-----------------------------------------
    case 3:
        switch (this->GetNameMinor())
        {
        case 1:                             //Name_Bior_3_1
            coeff.push_back((-1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            break;
        case 3:                             //Name_Bior_3_3
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back((-1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            break;
        case 5:                             //Name_Bior_3_5
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back((-1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            break;
        case 7:                             //Name_Bior_3_7
            //TODO:
            std::cerr << "ERROR: Bior3.7 is NOT YET IMPLEMENTED" << std::endl;
            break;
        case 9:                             //Name_Bior_3_9
            //TODO:
            std::cerr << "ERROR: Bior3.9 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)         
        break;
    //-----------------------------------------
    case 4:
        switch (this->GetNameMinor())
        {
        case 4:                             //name_Bior_4_4
            //TODO:
            std::cerr << "ERROR: Bior4.4 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)
        break;
    //-----------------------------------------
    case 5:
        switch (this->GetNameMinor())
        {
        case 5:                             //Name_Bior_5_5
            //TODO:
            std::cerr << "ERROR: Bior5.5 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)         
        break;
    //-----------------------------------------
    case 6:
        switch (this->GetNameMinor())
        {
        case 8:                             //Name_Bior_6_8
            //TODO:
            std::cerr << "ERROR: Bior6.8 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)
        break;  
    } //end case(nameMajor)

    return coeff;
}

template<class TPixel, unsigned int VDimension, class TAllocator>
typename BiorthogonalWaveletOperator<TPixel, VDimension, TAllocator>::
CoefficientVector 
BiorthogonalWaveletOperator<TPixel, VDimension, TAllocator>
::GenerateCoefficientsLowpassReconstruct()
{
    CoefficientVector coeff;
    
    switch (this->GetNameMajor())
    {
    //-----------------------------------------
    case 1:
        switch (this->GetNameMinor())
        {
        case 1:                             //Name_Bior_1_1
            coeff.push_back(1.0/vnl_math::sqrt2);
            coeff.push_back(1.0/vnl_math::sqrt2);
            break;
        case 3:                             //name_Bior_1_3
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(1.0/vnl_math::sqrt2);
            coeff.push_back(1.0/vnl_math::sqrt2);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            break;
        case 5:                             //name_Bior_1_5
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(1.0/vnl_math::sqrt2);
            coeff.push_back(1.0/vnl_math::sqrt2);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            break;
        } //end case(nameMinor)
        break;
    //-----------------------------------------
    case 2:
        switch (this->GetNameMinor())
        {
        case 2:                             //Name_Bior_2_2
            coeff.push_back(0.0);
            coeff.push_back((1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/2.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            break;
        case 4:                             //Name_Bior_2_4
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back((1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/2.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            break;
        case 6:                             //Name_Bior_2_6
            //TODO:
            std::cerr << "ERROR: Bior2.6 is NOT YET IMPLEMENTED" << std::endl;

            break;
        case 8:                             //Name_Bior_2_8
            //TODO:
            std::cerr << "ERROR: Bior2.8 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)
        break;
    //-----------------------------------------
    case 3:
        switch (this->GetNameMinor())
        {
        case 1:                             //Name_Bior_3_1
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            break;
        case 3:                             //Name_Bior_3_3
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            break;
        case 5:                             //Name_Bior_3_5
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            coeff.push_back(0.0);
            break;
        case 7:                             //Name_Bior_3_7
            //TODO:
            std::cerr << "ERROR: Bior3.7 is NOT YET IMPLEMENTED" << std::endl;
            break;
        case 9:                             //Name_Bior_3_9
            //TODO:
            std::cerr << "ERROR: Bior3.9 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)         
        break;
    //-----------------------------------------
    case 4:
        switch (this->GetNameMinor())
        {
        case 4:                             //name_Bior_4_4
            //TODO:
            std::cerr << "ERROR: Bior4.4 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)
        break;
    //-----------------------------------------
    case 5:
        switch (this->GetNameMinor())
        {
        case 5:                             //Name_Bior_5_5
            //TODO:
            std::cerr << "ERROR: Bior5.5 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)         
        break;
    //-----------------------------------------
    case 6:
        switch (this->GetNameMinor())
        {
        case 8:                             //Name_Bior_6_8
            //TODO:
            std::cerr << "ERROR: Bior6.8 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)
        break;  
    } //end case(nameMajor)

    return coeff;
}

template<class TPixel, unsigned int VDimension, class TAllocator>
typename BiorthogonalWaveletOperator<TPixel, VDimension, TAllocator>::
CoefficientVector 
BiorthogonalWaveletOperator<TPixel, VDimension, TAllocator>
::GenerateCoefficientsHighpassReconstruct()
{
    CoefficientVector coeff;
    
    switch (this->GetNameMajor())
    {
    //-----------------------------------------
    case 1:
        switch (this->GetNameMinor())
        {
        case 1:                             //Name_Bior_1_1
            coeff.push_back(1.0/vnl_math::sqrt2);
            coeff.push_back(-1.0/vnl_math::sqrt2);
            break;
        case 3:                             //name_Bior_1_3
            coeff.push_back((-1.0/16.0)*vnl_math::sqrt2);
            coeff.push_back((-1.0/16.0)*vnl_math::sqrt2);
            coeff.push_back(1.0/vnl_math::sqrt2);
            coeff.push_back(-1.0/vnl_math::sqrt2);
            coeff.push_back((1.0/16.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/16.0)*vnl_math::sqrt2);
            break;
        case 5:                             //name_Bior_1_5
            coeff.push_back((3.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((-11.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((-11.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/2.0)*vnl_math::sqrt2);
            coeff.push_back((-1.0/2.0)*vnl_math::sqrt2);
            coeff.push_back((11.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((11.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/256.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/256.0)*vnl_math::sqrt2);
            break;
        } //end case(nameMinor)
        break;
    //-----------------------------------------
    case 2:
        switch (this->GetNameMinor())
        {
        case 2:                             //Name_Bior_2_2
            coeff.push_back(0.0);
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            break;
        case 4:                             //Name_Bior_2_4
            coeff.push_back(0.0);
            coeff.push_back((-3.0/128.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((19.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((-45.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((19.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((1.0/8.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/128.0)*vnl_math::sqrt2);
            break;
        case 6:                             //Name_Bior_2_6
            //TODO:
            std::cerr << "ERROR: Bior2.6 is NOT YET IMPLEMENTED" << std::endl;
            break;
        case 8:                             //Name_Bior_2_8
            //TODO:
            std::cerr << "ERROR: Bior2.6 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)
        break;
    //-----------------------------------------
    case 3:
        switch (this->GetNameMinor())
        {
        case 1:                             //Name_Bior_3_1
            coeff.push_back((-1.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((3.0/4.0)*vnl_math::sqrt2);
            coeff.push_back((-1.0/4.0)*vnl_math::sqrt2);
            break;
        case 3:                             //Name_Bior_3_3
            coeff.push_back((3.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((9.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((-7.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((-45.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((45.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((7.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((-9.0/64.0)*vnl_math::sqrt2);
            coeff.push_back((-3.0/64.0)*vnl_math::sqrt2);
            break;
        case 5:                             //Name_Bior_3_5
            coeff.push_back((-5.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((-15.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((19.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((97.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((-26.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((-350.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((350.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((26.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((-97.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((-19.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((15.0/512.0)*vnl_math::sqrt2);
            coeff.push_back((5.0/512.0)*vnl_math::sqrt2);
            break;
        case 7:                             //Name_Bior_3_7
            //TODO:
            std::cerr << "ERROR: Bior3.7 is NOT YET IMPLEMENTED" << std::endl;
            break;
        case 9:                             //Name_Bior_3_9
            //TODO:
            std::cerr << "ERROR: Bior3.9 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)         
        break;
    //-----------------------------------------
    case 4:
        switch (this->GetNameMinor())
        {
        case 4:                             //name_Bior_4_4
            //TODO:
            std::cerr << "ERROR: Bior4.4 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)
        break;
    //-----------------------------------------
    case 5:
        switch (this->GetNameMinor())
        {
        case 5:                             //Name_Bior_5_5
            //TODO:
            std::cerr << "ERROR: Bior5.5 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)         
        break;
    //-----------------------------------------
    case 6:
        switch (this->GetNameMinor())
        {
        case 8:                             //Name_Bior_6_8
            //TODO:
            std::cerr << "ERROR: Bior6.8 is NOT YET IMPLEMENTED" << std::endl;
            break;
        } //end case(nameMinor)
        break;  
    } //end case(nameMajor)

    return coeff;
}


}// end namespace gift

#endif
