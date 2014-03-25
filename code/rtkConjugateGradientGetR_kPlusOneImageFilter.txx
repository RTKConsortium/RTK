#ifndef __rtkConjugateGradientGetR_kPlusOneImageFilter_txx
#define __rtkConjugateGradientGetR_kPlusOneImageFilter_txx

#include "rtkConjugateGradientGetR_kPlusOneImageFilter.h"

#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

namespace rtk
{

template< typename TInputType>
ConjugateGradientGetR_kPlusOneImageFilter<TInputType>::ConjugateGradientGetR_kPlusOneImageFilter()
{
    this->SetNumberOfRequiredInputs(3);
}

template< typename TInputType>
void ConjugateGradientGetR_kPlusOneImageFilter<TInputType>::SetRk(const TInputType* Rk)
{
    this->SetNthInput(0, const_cast<TInputType*>(Rk));
}

template< typename TInputType>
void ConjugateGradientGetR_kPlusOneImageFilter<TInputType>::SetPk(const TInputType* Pk)
{
    this->SetNthInput(1, const_cast<TInputType*>(Pk));
}

template< typename TInputType>
void ConjugateGradientGetR_kPlusOneImageFilter<TInputType>::SetAPk(const TInputType* APk)
{
    this->SetNthInput(2, const_cast<TInputType*>(APk));
}

template< typename TInputType>
typename TInputType::Pointer ConjugateGradientGetR_kPlusOneImageFilter<TInputType>::GetRk()
{
    return static_cast< TInputType * >
            ( this->itk::ProcessObject::GetInput(0) );
}

template< typename TInputType>
typename TInputType::Pointer ConjugateGradientGetR_kPlusOneImageFilter<TInputType>::GetPk()
{
    return static_cast< TInputType * >
            ( this->itk::ProcessObject::GetInput(1) );
}

template< typename TInputType>
typename TInputType::Pointer ConjugateGradientGetR_kPlusOneImageFilter<TInputType>::GetAPk()
{
    return static_cast< TInputType * >
            ( this->itk::ProcessObject::GetInput(2) );
}

template< typename TInputType>
void ConjugateGradientGetR_kPlusOneImageFilter<TInputType>
::GenerateData()
{
    float eps=1e-8;

    // Get the largest possible region in input
    typename TInputType::RegionType Largest = this->GetRk()->GetLargestPossibleRegion();

    // Get the output pointer
    TInputType *outputPtr = this->GetOutput();
    outputPtr->SetRegions(Largest);
    outputPtr->Allocate();
    outputPtr->FillBuffer(1);

    // Prepare iterators
    typedef itk::ImageRegionIterator<TInputType> RegionIterator;

    // Compute Norm(r_k)²
    float Norm_r_k_square = 0;
    RegionIterator r_k_It(this->GetRk(), Largest);
    r_k_It.GoToBegin();
    while(!r_k_It.IsAtEnd()){
        Norm_r_k_square += r_k_It.Get() * r_k_It.Get();
        ++r_k_It;
    }

    // Compute p_k_t_A_p_k
    float p_k_t_A_p_k = 0;
    RegionIterator p_k_It(this->GetPk(), Largest);
    p_k_It.GoToBegin();
    RegionIterator A_p_k_It(this->GetAPk(), Largest);
    A_p_k_It.GoToBegin();
    while(!p_k_It.IsAtEnd()){
        p_k_t_A_p_k += p_k_It.Get() * A_p_k_It.Get();
        ++p_k_It;
        ++A_p_k_It;
    }

    // Compute alpha_k
    m_alphak = Norm_r_k_square / (p_k_t_A_p_k + eps);

//    // Print m_alphak and p_k_t_A_p_k to detect potential division by zero problems
//    std::cout << "p_k_t_A_p_k = " << p_k_t_A_p_k << std::endl;
//    std::cout << "m_alphak = " << m_alphak << std::endl;

    // Compute Rk+1
    RegionIterator outputIt(outputPtr, Largest);
    outputIt.GoToBegin();
    A_p_k_It.GoToBegin();
    r_k_It.GoToBegin();
    while(!outputIt.IsAtEnd()){
        outputIt.Set(r_k_It.Get() - m_alphak * A_p_k_It.Get()) ;
        ++r_k_It;
        ++A_p_k_It;
        ++outputIt;
    }

}

}// end namespace


#endif
