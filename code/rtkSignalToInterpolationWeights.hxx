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
#ifndef rtkSignalToInterpolationWeights_hxx
#define rtkSignalToInterpolationWeights_hxx

#include "rtkSignalToInterpolationWeights.h"

#include <itksys/SystemTools.hxx>
#include <vcl_limits.h>
#include <itkMath.h>

namespace rtk
{

SignalToInterpolationWeights::SignalToInterpolationWeights()
{
  this->m_NumberOfReconstructedFrames = 0;
}

void SignalToInterpolationWeights::PrintSelf(std::ostream & os, itk::Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << this->m_Array2D << std::endl;
}

void SignalToInterpolationWeights::SetSignal(const std::vector<double> signal)
{
  this->m_Signal = signal;
}

/** Update method */
void SignalToInterpolationWeights::Update()
{
  // Set the matrix size
  unsigned int NumberOfProjections = m_Signal.size();
  this->m_Array2D.SetSize(this->m_NumberOfReconstructedFrames, NumberOfProjections);

  // Create a vector to hold the projections' Signal
  std::vector<float> reconstructedFrames; //Stores the instant of the cycle each frame represents

  // Compute the instant of the cycle each phase represents
  // Create one more phase point than is strictly necessary, so that projections Signal
  // are always between two reconstructed Signal. The first and last reconstructed Signal
  // represent the same instant. This is accounted for when computing the weights
  for (float n = 0.; n < this->m_NumberOfReconstructedFrames + 1; n++)
    reconstructedFrames.push_back(n/this->m_NumberOfReconstructedFrames);

  // Compute the interpolation weights and fill the Array2D
  m_Array2D.Fill(0);
  for (unsigned int c = 0; c < NumberOfProjections; c++)
    {
    int lower = 0; int upper = 1;
    while(!((m_Signal[c] >= reconstructedFrames[lower]) && (m_Signal[c] < reconstructedFrames[upper])))
      {
      lower++;
      upper++;
      if(lower == this->m_NumberOfReconstructedFrames)
        {
        std::cout << "Problem while determining the interpolation weights" << std::endl;
        }
      }
    float lowerWeight = (reconstructedFrames[upper] - m_Signal[c]) / (reconstructedFrames[upper] - reconstructedFrames[lower]);
    float upperWeight = (m_Signal[c] - reconstructedFrames[lower]) / (reconstructedFrames[upper] - reconstructedFrames[lower]);

    // The last phase is equal to the first one (see comment above when filling "reconstructedFrames")
    if (upper==this->m_NumberOfReconstructedFrames) upper=0;

    m_Array2D[lower][c] = itk::Math::Round<float>(lowerWeight * 100.) / 100.;
    m_Array2D[upper][c] = itk::Math::Round<float>(upperWeight * 100.) / 100.;
    }
}

/** Get the output */

SignalToInterpolationWeights::Array2DType
SignalToInterpolationWeights
::GetOutput()
{
  return this->m_Array2D;
}


} //end namespace rtk

#endif
