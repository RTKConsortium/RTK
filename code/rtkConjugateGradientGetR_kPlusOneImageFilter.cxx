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

#include "rtkConjugateGradientGetR_kPlusOneImageFilter.h"

namespace rtk
{

template<>
void
ConjugateGradientGetR_kPlusOneImageFilter<itk::VectorImage<float, 3>>
::ThreadedGenerateData(const itk::VectorImage<float, 3>::RegionType &
                           outputRegionForThread,
                           ThreadIdType threadId)
{
    double eps=1e-8;
    typedef itk::VectorImage<float, 3> TInputType;

    // Prepare iterators
    typedef itk::ImageRegionIterator<TInputType> RegionIterator;

    // Declare the correct pixel type and initialize its length (for vector images)
    typename TInputType::PixelType pix, Apix;
    unsigned int length = this->GetInput()->GetNumberOfComponentsPerPixel();

    // Compute Norm(r_k)²
    RegionIterator r_k_It(this->GetRk(), outputRegionForThread);
    r_k_It.GoToBegin();
    while(!r_k_It.IsAtEnd())
      {
      pix = r_k_It.Get();
      for (unsigned int comp = 0; comp<length; comp++)
        m_SquaredNormR_kVector[threadId] += pix[comp] * pix[comp];
      ++r_k_It;
      }

    // Compute p_k_t_A_p_k
    RegionIterator p_k_It(this->GetPk(), outputRegionForThread);
    p_k_It.GoToBegin();
    RegionIterator A_p_k_It(this->GetAPk(), outputRegionForThread);
    A_p_k_It.GoToBegin();
    while(!p_k_It.IsAtEnd())
      {
      pix = p_k_It.Get();
      Apix = A_p_k_It.Get();
      for (unsigned int comp = 0; comp<length; comp++)
        m_PktApkVector[threadId] += pix[comp] * Apix[comp];
      ++p_k_It;
      ++A_p_k_It;
      }
    m_Barrier->Wait();

    // Each thread computes alpha_k
    double squaredNormR_k = 0;
    double p_k_t_A_p_k = 0;
    for (unsigned int i=0; i<this->GetNumberOfThreads(); i++)
      {
      squaredNormR_k += m_SquaredNormR_kVector[i];
      p_k_t_A_p_k += m_PktApkVector[i];
      }
    double alphak = squaredNormR_k / (p_k_t_A_p_k + eps);

    // Compute Rk+1 and write it on the output
    RegionIterator outputIt(this->GetOutput(), outputRegionForThread);
    outputIt.GoToBegin();
    A_p_k_It.GoToBegin();
    r_k_It.GoToBegin();
    while(!outputIt.IsAtEnd())
      {
      pix = r_k_It.Get() - alphak * A_p_k_It.Get();
      outputIt.Set(pix);
      for (unsigned int comp = 0; comp<length; comp++)
        m_SquaredNormR_kPlusOneVector[threadId] += pix[comp] * pix[comp];
      ++r_k_It;
      ++A_p_k_It;
      ++outputIt;
      }
}

} // end namespace rtk
