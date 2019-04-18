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

#ifndef rtkConjugateGradientGetR_kPlusOneImageFilter_hxx
#define rtkConjugateGradientGetR_kPlusOneImageFilter_hxx

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
  this->AllocateOutputs();

  // Prepare iterators
  using RegionIterator = itk::ImageRegionIterator<TInputType>;

#if ITK_VERSION_MAJOR>4
  std::mutex accumulationLock;
#endif
  m_SquaredNormR_k = 0;
  double p_k_t_A_p_k = 0;
#if ITK_VERSION_MAJOR>4
  this->GetMultiThreader()->SetNumberOfWorkUnits( this->GetNumberOfWorkUnits() );
  this->GetMultiThreader()->template ParallelizeImageRegion<TInputType::ImageDimension>
    (
    this->GetOutput()->GetRequestedRegion(),
    [this, &p_k_t_A_p_k, &accumulationLock](const typename TInputType::RegionType & outputRegionForThread)
      {
#else
      {
      typename TInputType::RegionType outputRegionForThread;
      outputRegionForThread = this->GetOutput()->GetRequestedRegion();
#endif
      // Compute Norm(r_k)²
#if ITK_VERSION_MAJOR>4
      double squaredNormR_kThread = 0.;
#endif
      RegionIterator r_k_It(this->GetRk(), outputRegionForThread);
      r_k_It.GoToBegin();
      while(!r_k_It.IsAtEnd())
        {
#if ITK_VERSION_MAJOR>4
        squaredNormR_kThread += r_k_It.Get() * r_k_It.Get();
#else
        this->m_SquaredNormR_k += r_k_It.Get() * r_k_It.Get();
#endif
        ++r_k_It;
        }

      // Compute p_k_t_A_p_k
#if ITK_VERSION_MAJOR>4
      double p_k_t_A_p_kThread = 0.;
#endif
      RegionIterator p_k_It(this->GetPk(), outputRegionForThread);
      p_k_It.GoToBegin();
      RegionIterator A_p_k_It(this->GetAPk(), outputRegionForThread);
      A_p_k_It.GoToBegin();
      while(!p_k_It.IsAtEnd())
        {
#if ITK_VERSION_MAJOR>4
        p_k_t_A_p_kThread += p_k_It.Get() * A_p_k_It.Get();
#else
        p_k_t_A_p_k += p_k_It.Get() * A_p_k_It.Get();
#endif
        ++p_k_It;
        ++A_p_k_It;
        }
#if ITK_VERSION_MAJOR>4
      std::lock_guard<std::mutex> mutexHolder(accumulationLock);
      this->m_SquaredNormR_k += squaredNormR_kThread;
      p_k_t_A_p_k += p_k_t_A_p_kThread;
      },
    nullptr
    );
#else
      }
#endif

  const double eps=1e-8;
  typename itk::PixelTraits<typename TInputType::PixelType>::ValueType alphak = m_SquaredNormR_k / (p_k_t_A_p_k + eps);

  m_SquaredNormR_kPlusOne = 0;
#if ITK_VERSION_MAJOR>4
  this->GetMultiThreader()->SetNumberOfWorkUnits( this->GetNumberOfWorkUnits() );
  this->GetMultiThreader()->template ParallelizeImageRegion<TInputType::ImageDimension>
    (
    this->GetOutput()->GetRequestedRegion(),
    [this, alphak, &accumulationLock](const typename TInputType::RegionType & outputRegionForThread)
      {
#else
      {
      typename TInputType::RegionType outputRegionForThread;
      outputRegionForThread = this->GetOutput()->GetRequestedRegion();
#endif
      // Compute Rk+1 and write it on the output
#if ITK_VERSION_MAJOR>4
      double squaredNormR_kPlusOneVectorThread = 0.;
#endif
      RegionIterator outputIt(this->GetOutput(), outputRegionForThread);
      outputIt.GoToBegin();
      RegionIterator A_p_k_It(this->GetAPk(), outputRegionForThread);
      A_p_k_It.GoToBegin();
      RegionIterator r_k_It(this->GetRk(), outputRegionForThread);
      r_k_It.GoToBegin();
      while(!outputIt.IsAtEnd())
        {
        outputIt.Set(r_k_It.Get() - alphak * A_p_k_It.Get());
#if ITK_VERSION_MAJOR>4
        squaredNormR_kPlusOneVectorThread += outputIt.Get() * outputIt.Get();
#else
        this->m_SquaredNormR_kPlusOne += outputIt.Get() * outputIt.Get();
#endif
        ++r_k_It;
        ++A_p_k_It;
        ++outputIt;
        }
#if ITK_VERSION_MAJOR>4
      std::lock_guard<std::mutex> mutexHolder(accumulationLock);
      this->m_SquaredNormR_kPlusOne += squaredNormR_kPlusOneVectorThread;
      },
    nullptr
    );
#else
      }
#endif

  m_Alphak = m_SquaredNormR_k / (p_k_t_A_p_k + eps);
}

}// end namespace


#endif
