#ifndef __rtkConjugateGradientGetX_kPlusOneImageFilter_txx
#define __rtkConjugateGradientGetX_kPlusOneImageFilter_txx

#include "rtkConjugateGradientGetX_kPlusOneImageFilter.h"

namespace rtk
{

template< typename TInputType>
ConjugateGradientGetX_kPlusOneImageFilter<TInputType>::ConjugateGradientGetX_kPlusOneImageFilter()
{
  // Create the subfilters
  m_MultiplyFilter = MultiplyFilterType::New();
  m_AddFilter = AddFilterType::New();

  // Create a permanent connection
  m_AddFilter->SetInput1(m_MultiplyFilter->GetOutput());

  this->SetNumberOfRequiredInputs(2);
}

template< typename TInputType>
void ConjugateGradientGetX_kPlusOneImageFilter<TInputType>::SetXk(const TInputType* Xk)
{
    this->SetNthInput(0, const_cast<TInputType*>(Xk));
}

template< typename TInputType>
void ConjugateGradientGetX_kPlusOneImageFilter<TInputType>::SetPk(const TInputType* Pk)
{
    this->SetNthInput(1, const_cast<TInputType*>(Pk));
}

template< typename TInputType>
typename TInputType::Pointer ConjugateGradientGetX_kPlusOneImageFilter<TInputType>::GetXk()
{
    return static_cast< TInputType * >
            ( this->itk::ProcessObject::GetInput(0) );
}

template< typename TInputType>
typename TInputType::Pointer ConjugateGradientGetX_kPlusOneImageFilter<TInputType>::GetPk()
{
    return static_cast< TInputType * >
            ( this->itk::ProcessObject::GetInput(1) );
}

template< typename TInputType>
void
ConjugateGradientGetX_kPlusOneImageFilter<TInputType>
::GenerateOutputInformation()
{
  // Set inputs
  m_MultiplyFilter->SetInput1(this->GetPk());
  m_MultiplyFilter->SetConstant2(this->m_alphak);
  m_AddFilter->SetInput2(this->GetXk());

  // Have the last filter calculate its output information
  m_AddFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_AddFilter->GetOutput() );
}


template< typename TInputType>
void ConjugateGradientGetX_kPlusOneImageFilter<TInputType>
::GenerateData()
{
  // Set inputs
  m_MultiplyFilter->SetInput1(this->GetPk());
  m_MultiplyFilter->SetConstant2(this->m_alphak);
  m_AddFilter->SetInput2(this->GetXk());

  // Run the pipeline
  m_AddFilter->Update();

  // Pass the Add filter's output
  this->GraftOutput(m_AddFilter->GetOutput());
}

}// end namespace


#endif
