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

#ifndef rtkConjugateGradientGetX_kPlusOneImageFilter_hxx
#define rtkConjugateGradientGetX_kPlusOneImageFilter_hxx

#include "rtkConjugateGradientGetX_kPlusOneImageFilter.h"

namespace rtk
{

template< typename TInputType>
ConjugateGradientGetX_kPlusOneImageFilter<TInputType>::ConjugateGradientGetX_kPlusOneImageFilter()
{
  // Create the subfilters
  m_Alphak = 0;
  m_MultiplyFilter = MultiplyFilterType::New();
  m_AddFilter = AddFilterType::New();

  // Create a permanent connection
  m_AddFilter->SetInput1(m_MultiplyFilter->GetOutput());

  this->SetNumberOfRequiredInputs(2);

  // Set memory management options
  m_MultiplyFilter->ReleaseDataFlagOn();
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
  m_MultiplyFilter->SetConstant2(this->m_Alphak);
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
  // Run the pipeline
  m_AddFilter->Update();

  // Pass the Add filter's output
  this->GraftOutput(m_AddFilter->GetOutput());

  // Release data in internal filters
  m_MultiplyFilter->GetOutput()->ReleaseData();
}

}// end namespace


#endif
