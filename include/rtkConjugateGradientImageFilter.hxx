/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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

#include <itkMultiThreaderBase.h>
#include <mutex>
#include <itkIterationReporter.h>

namespace rtk
{

template <typename OutputImageType>
ConjugateGradientImageFilter<OutputImageType>::ConjugateGradientImageFilter()
{
  this->SetNumberOfRequiredInputs(2);
  this->SetInPlace(true);

  m_NumberOfIterations = 1;

  m_A = ConjugateGradientOperatorType::New();
}

template <typename OutputImageType>
void
ConjugateGradientImageFilter<OutputImageType>::SetX(const OutputImageType * OutputImage)
{
  this->SetNthInput(0, const_cast<OutputImageType *>(OutputImage));
}

template <typename OutputImageType>
void
ConjugateGradientImageFilter<OutputImageType>::SetB(const OutputImageType * OutputImage)
{
  this->SetNthInput(1, const_cast<OutputImageType *>(OutputImage));
}

template <typename OutputImageType>
typename ConjugateGradientImageFilter<OutputImageType>::OutputImagePointer
ConjugateGradientImageFilter<OutputImageType>::GetX()
{
  return static_cast<OutputImageType *>(this->itk::ProcessObject::GetInput(0));
}

template <typename OutputImageType>
typename ConjugateGradientImageFilter<OutputImageType>::OutputImagePointer
ConjugateGradientImageFilter<OutputImageType>::GetB()
{
  return static_cast<OutputImageType *>(this->itk::ProcessObject::GetInput(1));
}

template <typename OutputImageType>
void
ConjugateGradientImageFilter<OutputImageType>::SetA(ConjugateGradientOperatorType * _arg)
{
  this->m_A = _arg;
  this->Modified();
}

template <typename OutputImageType>
void
ConjugateGradientImageFilter<OutputImageType>::GenerateInputRequestedRegion()
{
  // Input 0 is the X of AX=B
  typename Superclass::InputImagePointer inputPtr0 = const_cast<OutputImageType *>(this->GetInput(0));
  if (!inputPtr0)
    return;
  inputPtr0->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());

  // Input 1 is the B of AX=B
  typename Superclass::InputImagePointer inputPtr1 = const_cast<OutputImageType *>(this->GetInput(1));
  if (!inputPtr1)
    return;
  inputPtr1->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
}

template <typename OutputImageType>
void
ConjugateGradientImageFilter<OutputImageType>::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();

  // Initialization
  m_A->SetX(this->GetX());

  // Compute output information
  this->m_A->UpdateOutputInformation();
}

template <typename OutputImageType>
void
ConjugateGradientImageFilter<OutputImageType>::GenerateData()
{
  typename OutputImageType::RegionType largest = this->GetOutput()->GetLargestPossibleRegion();
  using DataType = typename itk::PixelTraits<typename OutputImageType::PixelType>::ValueType;

  // Create and allocate images
  auto Pk = OutputImageType::New();
  auto Rk = OutputImageType::New();
  Pk->SetRegions(largest);
  Rk->SetRegions(largest);
  this->GetOutput()->SetRegions(largest);
  Pk->Allocate();
  Rk->Allocate();
  this->GetOutput()->Allocate();
  Pk->CopyInformation(this->GetOutput());
  Rk->CopyInformation(this->GetOutput());

  // In rtkConjugateGradientConeBeamReconstructionFilter, B is not updated
  // So at this point, it is only an empty shell. Let's update it
  this->GetB()->Update();
  m_A->Update();

  // Declare intermediate variables
  DataType numerator, denominator, alpha, beta;
  DataType eps = itk::NumericTraits<DataType>::min();

  // Instantiate the multithreader
  auto       mt = itk::MultiThreaderBase::New();
  std::mutex accumulationLock;

  // Compute Xk+1
  mt->template ParallelizeImageRegion<OutputImageType::ImageDimension>(
    largest,
    [this, Pk, Rk](const typename OutputImageType::RegionType & outputRegionForThread) {
      itk::ImageRegionIterator<OutputImageType> itP(Pk, outputRegionForThread);
      itk::ImageRegionIterator<OutputImageType> itR(Rk, outputRegionForThread);
      itk::ImageRegionIterator<OutputImageType> itB(this->GetB(), outputRegionForThread);
      itk::ImageRegionIterator<OutputImageType> itA_out(this->m_A->GetOutput(), outputRegionForThread);
      itk::ImageRegionIterator<OutputImageType> itIn(this->GetX(), outputRegionForThread);
      itk::ImageRegionIterator<OutputImageType> itX(this->GetOutput(), outputRegionForThread);
      while (!itP.IsAtEnd())
      {
        itR.Set(itB.Get() - itA_out.Get());
        itP.Set(itR.Get());
        itX.Set(itIn.Get());
        ++itP;
        ++itR;
        ++itA_out;
        ++itB;
        ++itIn;
        ++itX;
      }
    },
    nullptr);

  itk::IterationReporter iterationReporter(this, 0, 1);
  bool                   stopIterations = false;
  for (int iter = 0; (iter < m_NumberOfIterations) && !stopIterations; iter++)
  {
    // Compute A * Pk
    m_A->SetX(Pk);
    m_A->Update();

    // Compute alpha
    numerator = 0;
    denominator = 0;
    mt->template ParallelizeImageRegion<OutputImageType::ImageDimension>(
      largest,
      [this, Pk, Rk, &numerator, &denominator, &accumulationLock](
        const typename OutputImageType::RegionType & outputRegionForThread) {
        itk::ImageRegionIterator<OutputImageType> itP(Pk, outputRegionForThread);
        itk::ImageRegionIterator<OutputImageType> itR(Rk, outputRegionForThread);
        itk::ImageRegionIterator<OutputImageType> itA_out(this->m_A->GetOutput(), outputRegionForThread);
        DataType                                  currentThreadNumerator = 0.0;
        DataType                                  currentThreadDenominator = 0.0;
        while (!itP.IsAtEnd())
        {
          currentThreadNumerator += itR.Get() * itR.Get();
          currentThreadDenominator += itP.Get() * itA_out.Get();
          ++itR;
          ++itA_out;
          ++itP;
        }

        std::lock_guard<std::mutex> mutexHolder(accumulationLock);
        numerator += currentThreadNumerator;
        denominator += currentThreadDenominator;
      },
      nullptr);
    alpha = numerator / (denominator + eps);

    // Compute Xk+1
    mt->template ParallelizeImageRegion<OutputImageType::ImageDimension>(
      largest,
      [this, alpha, Pk](const typename OutputImageType::RegionType & outputRegionForThread) {
        itk::ImageRegionIterator<OutputImageType> itP(Pk, outputRegionForThread);
        itk::ImageRegionIterator<OutputImageType> itX(this->GetOutput(), outputRegionForThread);
        while (!itP.IsAtEnd())
        {
          itX.Set(itX.Get() + alpha * itP.Get());
          ++itX;
          ++itP;
        }
      },
      nullptr);

    // Compute Rk+1 and beta simultaneously
    denominator = numerator;
    numerator = 0;
    mt->template ParallelizeImageRegion<OutputImageType::ImageDimension>(
      largest,
      [this, Rk, &numerator, &accumulationLock, alpha](
        const typename OutputImageType::RegionType & outputRegionForThread) {
        itk::ImageRegionIterator<OutputImageType> itR(Rk, outputRegionForThread);
        itk::ImageRegionIterator<OutputImageType> itA_out(this->m_A->GetOutput(), outputRegionForThread);
        DataType                                  currentThreadNumerator = 0.0;
        while (!itR.IsAtEnd())
        {
          itR.Set(itR.Get() - alpha * itA_out.Get());
          currentThreadNumerator += itR.Get() * itR.Get();
          ++itR;
          ++itA_out;
        }

        std::lock_guard<std::mutex> mutexHolder(accumulationLock);
        numerator += currentThreadNumerator;
      },
      nullptr);
    beta = numerator / (denominator + eps);

    mt->template ParallelizeImageRegion<OutputImageType::ImageDimension>(
      largest,
      [Rk, Pk, beta](const typename OutputImageType::RegionType & outputRegionForThread) {
        itk::ImageRegionIterator<OutputImageType> itR(Rk, outputRegionForThread);
        itk::ImageRegionIterator<OutputImageType> itP(Pk, outputRegionForThread);
        while (!itR.IsAtEnd())
        {
          itP.Set(itR.Get() + beta * itP.Get());
          ++itR;
          ++itP;
        }
      },
      nullptr);

    // Let the m_A filter know that Pk has been modified, and it should
    // recompute its output at the beginning of next iteration
    Pk->Modified();
    iterationReporter.CompletedStep();
  }
  m_A->GetOutput()->ReleaseData();
}
} // namespace rtk


#endif
