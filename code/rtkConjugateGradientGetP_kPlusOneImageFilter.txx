#ifndef __rtkConjugateGradientGetP_kPlusOneImageFilter_txx
#define __rtkConjugateGradientGetP_kPlusOneImageFilter_txx

#include "rtkConjugateGradientGetP_kPlusOneImageFilter.h"

#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

namespace rtk
{

template< typename TInputType>
ConjugateGradientGetP_kPlusOneImageFilter<TInputType>::ConjugateGradientGetP_kPlusOneImageFilter()
{
    this->SetNumberOfRequiredInputs(3);
}

template< typename TInputType>
void ConjugateGradientGetP_kPlusOneImageFilter<TInputType>::SetR_kPlusOne(const TInputType* R_kPlusOne)
{
    this->SetNthInput(0, const_cast<TInputType*>(R_kPlusOne));
}

template< typename TInputType>
void ConjugateGradientGetP_kPlusOneImageFilter<TInputType>::SetRk(const TInputType* Rk)
{
    this->SetNthInput(1, const_cast<TInputType*>(Rk));
}

template< typename TInputType>
void ConjugateGradientGetP_kPlusOneImageFilter<TInputType>::SetPk(const TInputType* Pk)
{
    this->SetNthInput(2, const_cast<TInputType*>(Pk));
}

template< typename TInputType>
typename TInputType::Pointer ConjugateGradientGetP_kPlusOneImageFilter<TInputType>::GetR_kPlusOne()
{
    return static_cast< TInputType * >
            ( this->itk::ProcessObject::GetInput(0) );
}

template< typename TInputType>
typename TInputType::Pointer ConjugateGradientGetP_kPlusOneImageFilter<TInputType>::GetRk()
{
    return static_cast< TInputType * >
            ( this->itk::ProcessObject::GetInput(1) );
}

template< typename TInputType>
typename TInputType::Pointer ConjugateGradientGetP_kPlusOneImageFilter<TInputType>::GetPk()
{
    return static_cast< TInputType * >
            ( this->itk::ProcessObject::GetInput(2) );
}

template< typename TInputType>
void ConjugateGradientGetP_kPlusOneImageFilter<TInputType>
::GenerateData()
{
    float eps=0.00000001;

    // Get the largest possible region in input
    typename TInputType::RegionType Largest = this->GetRk()->GetLargestPossibleRegion();

    // Get the output pointer
    TInputType *outputPtr = this->GetOutput();
    outputPtr->SetRegions(Largest);
    outputPtr->Allocate();

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

    // Compute Norm(r__kPlusOne)²
    float Norm_r__kPlusOne_square = 0;
    RegionIterator r__kPlusOne_It(this->GetR_kPlusOne(), Largest);
    r__kPlusOne_It.GoToBegin();
    while(!r__kPlusOne_It.IsAtEnd()){
        Norm_r__kPlusOne_square += r__kPlusOne_It.Get() * r__kPlusOne_It.Get();
        ++r__kPlusOne_It;
    }

    // Compute Betak
    float Betak = Norm_r__kPlusOne_square / (Norm_r_k_square + eps);

//    // Print Betak and Norm_r_k_square to detect potential division by zero problems
//    std::cout << "Norm_r_k_square = " << Norm_r_k_square << std::endl;
//    std::cout << "Betak = " << Betak << std::endl;

    // Compute Pk+1
    RegionIterator outputIt(outputPtr, Largest);
    RegionIterator p_k_It(this->GetPk(), Largest);
    outputIt.GoToBegin();
    r__kPlusOne_It.GoToBegin();
    p_k_It.GoToBegin();
    while(!outputIt.IsAtEnd()){
        outputIt.Set(r__kPlusOne_It.Get() + Betak * p_k_It.Get()) ;
        ++r__kPlusOne_It;
        ++p_k_It;
        ++outputIt;
    }

}

}// end namespace


#endif
