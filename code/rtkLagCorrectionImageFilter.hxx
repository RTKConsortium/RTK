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

#ifndef __rtkLagCorrectionImageFilter_hxx
#define __rtkLagCorrectionImageFilter_hxx

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include <iterator>

namespace rtk
{

template<typename TImage, unsigned ModelOrder>
LagCorrectionImageFilter<TImage, ModelOrder>::LagCorrectionImageFilter()
{
  this->SetNumberOfRequiredInputs(1);
  m_A.Fill(0.0f);
  m_B.Fill(0.0f);
  m_ExpmA.Fill(0.0f);
  m_BExpmA.Fill(0.0f);

  m_NewParamJustReceived = false;
  m_StatusMatrixAllocated = false;
}

template<typename TImage, unsigned ModelOrder>
void LagCorrectionImageFilter<TImage, ModelOrder>::BeforeThreadedGenerateData()
{
  // Initialization
  if (m_NewParamJustReceived) {
    int i = 0;
    typename VectorType::Iterator itB = m_B.Begin();
    for (typename VectorType::Iterator itA = m_A.Begin(); itA != m_A.End(); ++itA, ++itB, ++i) {
      float expma = expf(-*itA);
      m_ExpmA[i] = expma;
      m_BExpmA[i] = (*itB) * expma;
    }

    if (!m_StatusMatrixAllocated) {
      ImageSizeType SizeInput = this->GetInput()->GetLargestPossibleRegion().GetSize();
      m_M = SizeInput[0] * SizeInput[1];

      VectorType v;
      v.Fill(0.0f);
      m_S = StateType::New();
      m_S->SetRegions(this->GetInput()->GetLargestPossibleRegion());
      m_S->Allocate();
      m_S->FillBuffer(v);
      m_StatusMatrixAllocated = true;
    }

    m_ImageId = 0;
    m_NewParamJustReceived = false;
  }

  m_ThAvgCorr = 0.0f;
}

template<typename TImage, unsigned ModelOrder>
void LagCorrectionImageFilter<TImage, ModelOrder>::AfterThreadedGenerateData()
{
  m_AvgCorrections.push_back(m_ThAvgCorr / (float)(this->GetNumberOfThreads() * m_M));
  ++m_ImageId;
}

template<typename TImage, unsigned ModelOrder>
void LagCorrectionImageFilter<TImage, ModelOrder>::
  ThreadedGenerateData(const ImageRegionType & thRegion, ThreadIdType threadId)
{
  typename TImage::Pointer  inputPtr = const_cast<TImage *>(this->GetInput());
  typename TImage::Pointer outputPtr = this->GetOutput();

  typename TImage::RegionType reg = thRegion;
  typename TImage::IndexType start = thRegion.GetIndex();
  start[2] = 0;
  reg.SetIndex(start);

  itk::ImageRegionConstIterator<TImage> itIn(inputPtr, reg);
  itk::ImageRegionIterator<TImage>      itOut(outputPtr, reg);
  itk::ImageRegionIterator<StateType>   itS(m_S, reg);

  itIn.GoToBegin();
  itOut.GoToBegin();
  itS.GoToBegin();

  float meanc = 0.0f;      // Average correction over all projection
  while (!itIn.IsAtEnd())
  {
    VectorType S = itS.Get();

    // k is pixel id
    float c = 0.0f;
    for (unsigned int n = 0; n < ModelOrder; n++) {
      c += m_BExpmA[n] * S[n];
    }
    meanc += c;

    float xk = float(itIn.Get()) - c;
    /*if (xk<0.0f) {
      xk = 0.0f;
      } else if (xk >= 65536) {
      xk = 65535;
      }*/
    itOut.Set(static_cast<PixelType>(xk));

    // Update internal state Snk
    for (unsigned int n = 0; n < ModelOrder; n++) {
      S[n] = xk + m_ExpmA[n] * S[n];
    }
    itS.Set(S);

    ++itIn;
    ++itOut;
    ++itS;
  }

  m_Mutex.Lock();
  m_ThAvgCorr += meanc;
  m_Mutex.Unlock();
}

} // end namespace

#endif
