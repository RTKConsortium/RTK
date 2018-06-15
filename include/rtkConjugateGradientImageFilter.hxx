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

#ifndef rtkConjugateGradientImageFilter_hxx
#define rtkConjugateGradientImageFilter_hxx

#include <itkImageFileWriter.h>
#include "rtkConjugateGradientImageFilter.h"
#include <string>       // std::string
#include <sstream>      // std::stringstream

namespace rtk
{

template<typename OutputImageType>
ConjugateGradientImageFilter<OutputImageType>::ConjugateGradientImageFilter()
{
  this->SetNumberOfRequiredInputs(2);

  m_NumberOfIterations = 1;
//  m_TargetSumOfSquaresBetweenConsecutiveIterates = 0;
  m_IterationCosts = false;
  m_C=0.0;

  m_A = ConjugateGradientOperatorType::New();
}

template<typename OutputImageType>
void ConjugateGradientImageFilter<OutputImageType>
::SetC(const double _arg)
{
  this->m_C = _arg;
  this->Modified();
}

template<typename OutputImageType>
double ConjugateGradientImageFilter<OutputImageType>
::GetC()
{
  return this->m_C;
}

template<typename OutputImageType>
const std::vector<double> &ConjugateGradientImageFilter<OutputImageType>
::GetResidualCosts()
{
  return this->m_ResidualCosts;
}

template<typename OutputImageType>
void ConjugateGradientImageFilter<OutputImageType>::SetX(const OutputImageType* OutputImage)
{
  this->SetNthInput(0, const_cast<OutputImageType*>(OutputImage));
}

template<typename OutputImageType>
void ConjugateGradientImageFilter<OutputImageType>::SetB(const OutputImageType* OutputImage)
{
  this->SetNthInput(1, const_cast<OutputImageType*>(OutputImage));
}

template<typename OutputImageType>
typename ConjugateGradientImageFilter<OutputImageType>::OutputImagePointer
ConjugateGradientImageFilter<OutputImageType>::GetX()
{
  return static_cast< OutputImageType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template<typename OutputImageType>
typename ConjugateGradientImageFilter<OutputImageType>::OutputImagePointer
ConjugateGradientImageFilter<OutputImageType>::GetB()
{
  return static_cast< OutputImageType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template<typename OutputImageType>
void ConjugateGradientImageFilter<OutputImageType>
::SetA(ConjugateGradientOperatorPointerType _arg )
{
  this->m_A = _arg;
  this->Modified();
}

template<typename OutputImageType>
void ConjugateGradientImageFilter<OutputImageType>
::GenerateInputRequestedRegion()
{
  // Input 0 is the X of AX=B
  typename Superclass::InputImagePointer inputPtr0 =
    const_cast< OutputImageType * >( this->GetInput(0) );
  if ( !inputPtr0 ) return;
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the B of AX=B
  typename Superclass::InputImagePointer inputPtr1 =
    const_cast< OutputImageType * >( this->GetInput(1) );
  if ( !inputPtr1 ) return;
  inputPtr1->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );
}

template<typename OutputImageType>
void ConjugateGradientImageFilter<OutputImageType>
::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();

  // Initialization
  m_A->SetX(this->GetX());
  m_A->ReleaseDataFlagOn();

  // Compute output information
  this->m_A->UpdateOutputInformation();
}

template<typename OutputImageType>
void ConjugateGradientImageFilter<OutputImageType>
::GenerateData()
{
  typename OutputImageType::RegionType largest = this->GetOutput()->GetLargestPossibleRegion();

  // Create and allocate images
  typename OutputImageType::Pointer Pk = OutputImageType::New();
  typename OutputImageType::Pointer Rk = OutputImageType::New();
  Pk->SetRegions(largest);
  Rk->SetRegions(largest);
  this->GetOutput()->SetRegions(largest);
  Pk->Allocate();
  Rk->Allocate();
  this->GetOutput()->Allocate();

  // In rtkConjugateGradientConeBeamReconstructionFilter, B is not updated
  // So at this point, it is only an empty shell. Let's update it
  this->GetB()->Update();
  m_A->Update();

  // Declare many intermediate variables
  typename itk::PixelTraits<typename OutputImageType::PixelType>::ValueType numerator, denominator, alpha, beta;
  itk::ImageRegionConstIterator<OutputImageType> itB(this->GetB(), largest);
  itk::ImageRegionIterator<OutputImageType> itA_out;
  itk::ImageRegionIterator<OutputImageType> itP(Pk, largest);
  itk::ImageRegionIterator<OutputImageType> itR(Rk, largest);
  itk::ImageRegionIterator<OutputImageType> itX(this->GetOutput(), largest);
  itA_out = itk::ImageRegionIterator<OutputImageType>(m_A->GetOutput(), largest);

  // Initialize P0 and R0
  while(!itP.IsAtEnd())
    {
    itP.Set(itB.Get() - itA_out.Get());
    itR.Set(itB.Get() - itA_out.Get());
    ++itP;
    ++itR;
    ++itA_out;
    ++itB;
    }
  itP.GoToBegin();
  itR.GoToBegin();
  itA_out.GoToBegin();
  itB.GoToBegin();

  bool stopIterations = false;
  for(unsigned int iter=0; (iter<m_NumberOfIterations) && !stopIterations; iter++)
    {
    // Compute A * Pk
    m_A->SetX(Pk);
    m_A->Update();
    itA_out = itk::ImageRegionIterator<OutputImageType>(m_A->GetOutput(), largest);

    // Compute alpha
    numerator = 0;
    denominator = 0;
    while(!itP.IsAtEnd())
      {
      numerator += itR.Get() * itR.Get();
      denominator += itP.Get() * itA_out.Get();
      ++itR;
      ++itA_out;
      ++itP;
      }
    itP.GoToBegin();
    itR.GoToBegin();
    itA_out.GoToBegin();
    alpha = numerator / denominator;

    // Compute Xk+1
    while(!itP.IsAtEnd())
      {
      itX.Set(itX.Get() + alpha * itP.Get());
      ++itX;
      ++itP;
      }
    itP.GoToBegin();
    itX.GoToBegin();

    // Compute Rk+1 and beta simultaneously
    denominator = numerator;
    numerator = 0;
    while(!itR.IsAtEnd())
      {
      itR.Set(itR.Get() - alpha * itA_out.Get());
      numerator += itR.Get() * itR.Get();
      ++itR;
      ++itA_out;
      }
    itR.GoToBegin();
    itA_out.GoToBegin();
    beta = numerator / denominator;

    // Compute Pk+1
    while(!itP.IsAtEnd())
      {
      itP.Set(itR.Get() + beta * itP.Get());
      ++itR;
      ++itP;
      }
    itR.GoToBegin();
    itP.GoToBegin();
    }
}
}// end namespace


#endif
