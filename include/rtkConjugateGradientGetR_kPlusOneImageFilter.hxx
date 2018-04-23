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
ConjugateGradientGetR_kPlusOneImageFilter<TInputType>::ConjugateGradientGetR_kPlusOneImageFilter():
    m_Alphak(0.),
    m_SquaredNormR_k(0.),
    m_SquaredNormR_kPlusOne(0.)
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
::BeforeThreadedGenerateData()
{
  // Instead of using GetNumberOfThreads, we need to split the image into the
  // number of regions that will actually be returned by
  // itkImageSource::SplitRequestedRegion. Sometimes this number is less than
  // the number of threads requested.
  OutputImageRegionType dummy;
  unsigned int actualThreads = this->SplitRequestedRegion(
    0, this->GetNumberOfThreads(), dummy);

  m_Barrier = itk::Barrier::New();
  m_Barrier->Initialize(actualThreads);

  m_SquaredNormR_kVector.clear();
  m_SquaredNormR_kPlusOneVector.clear();
  m_PktApkVector.clear();

  for (unsigned int i=0; i<this->GetNumberOfThreads(); i++)
    {
    m_SquaredNormR_kVector.push_back(0);
    m_SquaredNormR_kPlusOneVector.push_back(0);
    m_PktApkVector.push_back(0);
    }
}

template< typename TInputType>
void ConjugateGradientGetR_kPlusOneImageFilter<TInputType>
::ThreadedGenerateData(const typename TInputType::RegionType & outputRegionForThread, ThreadIdType threadId)
{
  float eps=1e-8;

  // Prepare iterators
  typedef itk::ImageRegionIterator<TInputType> RegionIterator;

  // Compute Norm(r_k)²
  RegionIterator r_k_It(this->GetRk(), outputRegionForThread);
  r_k_It.GoToBegin();
  while(!r_k_It.IsAtEnd())
    {
    m_SquaredNormR_kVector[threadId] += r_k_It.Get() * r_k_It.Get();
    ++r_k_It;
    }

  // Compute p_k_t_A_p_k
  RegionIterator p_k_It(this->GetPk(), outputRegionForThread);
  p_k_It.GoToBegin();
  RegionIterator A_p_k_It(this->GetAPk(), outputRegionForThread);
  A_p_k_It.GoToBegin();
  while(!p_k_It.IsAtEnd())
    {
    m_PktApkVector[threadId] += p_k_It.Get() * A_p_k_It.Get();
    ++p_k_It;
    ++A_p_k_It;
    }
  m_Barrier->Wait();

  // Each thread computes alpha_k
  float squaredNormR_k = 0;
  float p_k_t_A_p_k = 0;
  for (unsigned int i=0; i<this->GetNumberOfThreads(); i++)
    {
    squaredNormR_k += m_SquaredNormR_kVector[i];
    p_k_t_A_p_k += m_PktApkVector[i];
    }
  float alphak = squaredNormR_k / (p_k_t_A_p_k + eps);

  // Compute Rk+1 and write it on the output
  RegionIterator outputIt(this->GetOutput(), outputRegionForThread);
  outputIt.GoToBegin();
  A_p_k_It.GoToBegin();
  r_k_It.GoToBegin();
  while(!outputIt.IsAtEnd())
    {
    outputIt.Set(r_k_It.Get() - alphak * A_p_k_It.Get());
    m_SquaredNormR_kPlusOneVector[threadId] += outputIt.Get() * outputIt.Get();
    ++r_k_It;
    ++A_p_k_It;
    ++outputIt;
    }
}

template< typename TInputType>
void ConjugateGradientGetR_kPlusOneImageFilter<TInputType>
::AfterThreadedGenerateData()
{
  float eps=1e-8;

  // Set the members m_Alphak, m_SquaredNormR_k and
  // m_SquaredNormR_kPlusOne, as they will be passed to other filters
  m_SquaredNormR_k = 0;
  m_SquaredNormR_kPlusOne = 0;
  float p_k_t_A_p_k = 0;
  for (unsigned int i=0; i<this->GetNumberOfThreads(); i++)
    {
    m_SquaredNormR_k += m_SquaredNormR_kVector[i];
    m_SquaredNormR_kPlusOne += m_SquaredNormR_kPlusOneVector[i];
    p_k_t_A_p_k += m_PktApkVector[i];
    }
  m_Alphak = m_SquaredNormR_k / (p_k_t_A_p_k + eps);
}

}// end namespace


#endif
