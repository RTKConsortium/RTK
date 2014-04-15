#ifndef _giftDaubechiesWaveletOperator_TXX
#define _giftDaubechiesWaveletOperator_TXX

//Includes
#include "giftDaubechiesWaveletOperator.h"
#include "vnl/vnl_math.h"
#include <vector>
#include <algorithm>

namespace gift
{

template<class TPixel, unsigned int VDimension, class TAllocator>
typename DaubechiesWaveletOperator<TPixel, VDimension, TAllocator>
::CoefficientVector
DaubechiesWaveletOperator<TPixel, VDimension, TAllocator>
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
typename DaubechiesWaveletOperator<TPixel, VDimension, TAllocator>::
CoefficientVector
DaubechiesWaveletOperator<TPixel, VDimension, TAllocator>
::GenerateCoefficientsLowpassReconstruct()
{
    CoefficientVector coeff;

    // IMPORTANT NOTE : The kernels are in reverse order with respect to
    // what can be found in the litterature. This is because ITK easily computes
    // the inner product between an image region and a kernel, but does not mirror
    // the kernel beforehand (which is necessary to perform a convolution). It is
    // compensated by defining the kernel as the mirrors of what they really are.

    switch (this->GetOrder())
    {
    case 1:
        coeff.push_back(1.0/vnl_math::sqrt2);
        coeff.push_back(1.0/vnl_math::sqrt2);
        break;
    case 2:
        coeff.push_back(-0.1830127/vnl_math::sqrt2);
        coeff.push_back(0.3169873/vnl_math::sqrt2);
        coeff.push_back(1.1830127/vnl_math::sqrt2);
        coeff.push_back(0.6830127/vnl_math::sqrt2);
        break;
    case 3:
        coeff.push_back(0.0498175/vnl_math::sqrt2);
        coeff.push_back(-0.12083221/vnl_math::sqrt2);
        coeff.push_back(-0.19093442/vnl_math::sqrt2);
        coeff.push_back(0.650365/vnl_math::sqrt2);
        coeff.push_back(1.14111692/vnl_math::sqrt2);
        coeff.push_back(0.47046721/vnl_math::sqrt2);
        break;
    case 4:
        coeff.push_back(-0.01498699/vnl_math::sqrt2);
        coeff.push_back(0.0465036/vnl_math::sqrt2);
        coeff.push_back(0.0436163/vnl_math::sqrt2);
        coeff.push_back(-0.26450717/vnl_math::sqrt2);
        coeff.push_back(-0.03957503/vnl_math::sqrt2);
        coeff.push_back(0.8922014/vnl_math::sqrt2);
        coeff.push_back(1.01094572/vnl_math::sqrt2);
        coeff.push_back(0.32580343/vnl_math::sqrt2);
        break;
    case 5:
        coeff.push_back(0.00471742793/vnl_math::sqrt2);
        coeff.push_back(-0.01779187/vnl_math::sqrt2);
        coeff.push_back(-0.00882680/vnl_math::sqrt2);
        coeff.push_back(0.10970265/vnl_math::sqrt2);
        coeff.push_back(-0.04560113/vnl_math::sqrt2);
        coeff.push_back(-0.34265671/vnl_math::sqrt2);
        coeff.push_back(0.19576696/vnl_math::sqrt2);
        coeff.push_back(1.02432694/vnl_math::sqrt2);
        coeff.push_back(0.85394354/vnl_math::sqrt2);
        coeff.push_back(0.22641898/vnl_math::sqrt2);
        break;
    case 6:
        coeff.push_back(-0.00152353381/vnl_math::sqrt2);
        coeff.push_back(0.00675606236/vnl_math::sqrt2);
        coeff.push_back(0.000783251152/vnl_math::sqrt2);
        coeff.push_back(-0.04466375/vnl_math::sqrt2);
        coeff.push_back(0.03892321/vnl_math::sqrt2);
        coeff.push_back(0.13788809/vnl_math::sqrt2);
        coeff.push_back(-0.18351806/vnl_math::sqrt2);
        coeff.push_back(-0.31998660/vnl_math::sqrt2);
        coeff.push_back(0.44583132/vnl_math::sqrt2);
        coeff.push_back(1.06226376/vnl_math::sqrt2);
        coeff.push_back(0.69950381/vnl_math::sqrt2);
        coeff.push_back(0.15774243/vnl_math::sqrt2);
        break;
    case 7:
        coeff.push_back(0.000500226853/vnl_math::sqrt2);
        coeff.push_back(-0.00254790472/vnl_math::sqrt2);
        coeff.push_back(0.000607514995/vnl_math::sqrt2);
        coeff.push_back(0.01774979/vnl_math::sqrt2);
        coeff.push_back(-0.02343994/vnl_math::sqrt2);
        coeff.push_back(-0.05378245/vnl_math::sqrt2);
        coeff.push_back(0.11400345/vnl_math::sqrt2);
        coeff.push_back(0.1008467/vnl_math::sqrt2);
        coeff.push_back(-0.31683501/vnl_math::sqrt2);
        coeff.push_back(-0.20351382/vnl_math::sqrt2);
        coeff.push_back(0.66437248/vnl_math::sqrt2);
        coeff.push_back(1.03114849/vnl_math::sqrt2);
        coeff.push_back(0.56079128/vnl_math::sqrt2);
        coeff.push_back(0.11009943/vnl_math::sqrt2);
        break;
    } //end case(Order)

    return coeff;
}

template<class TPixel, unsigned int VDimension, class TAllocator>
typename DaubechiesWaveletOperator<TPixel, VDimension, TAllocator>::
CoefficientVector
DaubechiesWaveletOperator<TPixel, VDimension, TAllocator>
::GenerateCoefficientsHighpassDeconstruct()
{
    CoefficientVector coeff = this->GenerateCoefficientsLowpassReconstruct();
    unsigned int it;

    double factor = 1;
    for (it=0; it<coeff.size(); it++)
    {
        coeff[it] *= factor;
        factor *= -1;
    }
    return coeff;
}

template<class TPixel, unsigned int VDimension, class TAllocator>
typename DaubechiesWaveletOperator<TPixel, VDimension, TAllocator>::
CoefficientVector
DaubechiesWaveletOperator<TPixel, VDimension, TAllocator>
::GenerateCoefficientsLowpassDeconstruct()
{


    CoefficientVector coeff = this->GenerateCoefficientsLowpassReconstruct();
    std::reverse(coeff.begin(), coeff.end());

// // Printing debug information
//    std::cout << "Generating low-pass deconstruction kernel" << std::endl;
//    unsigned int it;
//    for (it=0; it<coeff.size(); it++)
//    {
//        std::cout << "coeff["<< it << "] = " << coeff[it] << std::endl;
//    }

    return coeff;
}

template<class TPixel, unsigned int VDimension, class TAllocator>
typename DaubechiesWaveletOperator<TPixel, VDimension, TAllocator>::
CoefficientVector
DaubechiesWaveletOperator<TPixel, VDimension, TAllocator>
::GenerateCoefficientsHighpassReconstruct()
{
    CoefficientVector coeff = this->GenerateCoefficientsHighpassDeconstruct();
    std::reverse(coeff.begin(), coeff.end());
    return coeff;
}


}// end namespace gift

#endif
