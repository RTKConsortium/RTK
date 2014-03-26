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
  // Create the subfilters
  m_MultiplyFilter = MultiplyFilterType::New();
  m_AddFilter = AddFilterType::New();

  // Create a permanent connection
  m_AddFilter->SetInput1(m_MultiplyFilter->GetOutput());

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
void
ConjugateGradientGetP_kPlusOneImageFilter<TInputType>
::GenerateOutputInformation()
{
  // Compute m_Betak
  float eps=1e-8;
  m_Betak = m_SquaredNormR_kPlusOne / (m_SquaredNormR_k + eps);

  // Set inputs
  m_MultiplyFilter->SetInput1(this->GetPk());
  m_MultiplyFilter->SetConstant2(this->m_Betak);
  m_AddFilter->SetInput2(this->GetR_kPlusOne());

  // Have the last filter calculate its output information
  m_AddFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_AddFilter->GetOutput() );
}


template< typename TInputType>
void ConjugateGradientGetP_kPlusOneImageFilter<TInputType>
::GenerateData()
{
  // Run the pipeline
  m_AddFilter->Update();

  // Pass the Add filter's output
  this->GraftOutput(m_AddFilter->GetOutput());
}

}// end namespace


#endif
