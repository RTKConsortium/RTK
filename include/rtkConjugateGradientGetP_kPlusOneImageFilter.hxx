/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkConjugateGradientGetP_kPlusOneImageFilter_hxx
#define rtkConjugateGradientGetP_kPlusOneImageFilter_hxx

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

  m_SquaredNormR_k = 0;
  m_SquaredNormR_kPlusOne = 0;
  m_Betak = 0;

  this->SetNumberOfRequiredInputs(3);

  // Set memory management options
  m_MultiplyFilter->ReleaseDataFlagOn();
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

  // Release data in internal filters
  m_MultiplyFilter->GetOutput()->ReleaseData();
}

}// end namespace


#endif
