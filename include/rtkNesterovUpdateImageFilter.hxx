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

#ifndef rtkNesterovUpdateImageFilter_hxx
#define rtkNesterovUpdateImageFilter_hxx

#include "rtkNesterovUpdateImageFilter.h"
#include "itkImageRegionIterator.h"

namespace rtk
{

template<typename TImage>
NesterovUpdateImageFilter<TImage>::NesterovUpdateImageFilter()
{
  this->SetNumberOfRequiredInputs(2);
  m_Vk = TImage::New();
  m_Alphak = TImage::New();

  m_NumberOfIterations = 100;
  m_MustInitializeIntermediateImages = true;
  this->ResetIterations();
}

template<typename TImage>
NesterovUpdateImageFilter<TImage>::~NesterovUpdateImageFilter()
{
}

template<typename TImage>
void
NesterovUpdateImageFilter<TImage>::ResetIterations()
{
  // Nesterov's coefficients change at every iteration,
  // and this filter only performs one iteration. It needs
  // to keep track of the iteration number
  m_CurrentIteration = 0;

  // Precompute enough Nesterov coefficients for the expected
  // number of iterations
  m_tCoeffs.clear();
  m_Sums.clear();
  m_Ratios.clear();
  m_tCoeffs.push_back(1.0);
  m_Sums.push_back(0.0);
  m_Ratios.push_back(0.0);
  for (int k=1; k<m_NumberOfIterations; k++)
    {
    m_tCoeffs.push_back(0.5 * (1 + sqrt(1+ 4 * m_tCoeffs[k-1] * m_tCoeffs[k-1])));
    m_Sums.push_back(m_Sums[k-1] + m_tCoeffs[k]);
    m_Ratios.push_back(m_tCoeffs[k] / m_Sums[k]);
    }

  m_MustInitializeIntermediateImages = true;
}

template<typename TImage>
void NesterovUpdateImageFilter<TImage>
::GenerateInputRequestedRegion()
{
  // Input 0 is the image to update
  typename Superclass::InputImagePointer inputPtr0 =
    const_cast< TImage * >( this->GetInput(0) );
  if ( !inputPtr0 ) return;
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the update computed by Newton's method
  typename Superclass::InputImagePointer inputPtr1 =
    const_cast< TImage * >( this->GetInput(1) );
  if ( !inputPtr1 ) return;
  inputPtr1->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );
}

template<typename TImage>
void NesterovUpdateImageFilter<TImage>
::BeforeThreadedGenerateData()
{
  if (m_MustInitializeIntermediateImages)
    {
    // Allocate the intermediate images
    m_Vk->CopyInformation(this->GetInput(0));
    m_Vk->SetRegions(m_Vk->GetLargestPossibleRegion());
    m_Vk->Allocate();
    m_Alphak->CopyInformation(this->GetInput(0));
    m_Alphak->SetRegions(m_Alphak->GetLargestPossibleRegion());
    m_Alphak->Allocate();
    }
}

template<typename TImage>
void NesterovUpdateImageFilter<TImage>
#if ITK_VERSION_MAJOR<5
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
#else
::DynamicThreadedGenerateData(const OutputImageRegionType& outputRegionForThread)
#endif
{
  if (m_MustInitializeIntermediateImages)
    {
    // Copy the input 0 into them
    itk::ImageRegionConstIterator<TImage> inIt(this->GetInput(0), outputRegionForThread);
    itk::ImageRegionIterator<TImage> vIt(m_Vk, outputRegionForThread);
    itk::ImageRegionIterator<TImage> alphaIt(m_Alphak, outputRegionForThread);

    while(!inIt.IsAtEnd())
      {
      vIt.Set(inIt.Get());
      alphaIt.Set(inIt.Get());

      ++inIt;
      ++vIt;
      ++alphaIt;
      }
    }

  // Create iterators for all inputs and outputs
  itk::ImageRegionIterator<TImage> v_k_It(m_Vk, outputRegionForThread);
  itk::ImageRegionConstIterator<TImage> z_k_ItIn(this->GetInput(), outputRegionForThread);
  itk::ImageRegionIterator<TImage> z_k_ItOut(this->GetOutput(), outputRegionForThread);
  itk::ImageRegionConstIterator<TImage> g_k_It(this->GetInput(1), outputRegionForThread);

  // Perform computations
  if(m_CurrentIteration == m_NumberOfIterations)
    {
    // Last iteration, no need to update v_k and z_k. Return alpha_k, the current iterate.
    itk::ImageRegionIterator<TImage> alpha_k_It(this->GetOutput(), outputRegionForThread);
    while(!alpha_k_It.IsAtEnd())
      {
      alpha_k_It.Set(z_k_ItIn.Get() - g_k_It.Get());
      ++alpha_k_It;
      ++z_k_ItIn;
      ++g_k_It;
      }
    }
  else
    {
    itk::ImageRegionIterator<TImage> alpha_k_It(m_Alphak, outputRegionForThread);
    while(!alpha_k_It.IsAtEnd())
      {
      alpha_k_It.Set(z_k_ItIn.Get() - g_k_It.Get());
      v_k_It.Set(v_k_It.Get() - m_tCoeffs[m_CurrentIteration] * g_k_It.Get());
      z_k_ItOut.Set(alpha_k_It.Get() + m_Ratios[m_CurrentIteration + 1] * (v_k_It.Get() - alpha_k_It.Get()) );

      ++alpha_k_It;
      ++v_k_It;
      ++z_k_ItIn;
      ++z_k_ItOut;
      ++g_k_It;
      }
    }
}

template<typename TImage>
void NesterovUpdateImageFilter<TImage>
::AfterThreadedGenerateData()
{
  m_CurrentIteration++;
  m_MustInitializeIntermediateImages = false;
}

}// end namespace


#endif
