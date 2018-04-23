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
#ifndef rtkPhasesToInterpolationWeights_hxx
#define rtkPhasesToInterpolationWeights_hxx

#include "rtkPhasesToInterpolationWeights.h"

#include <itksys/SystemTools.hxx>
#include <vcl_limits.h>
#include <itkMath.h>

namespace rtk
{

PhasesToInterpolationWeights::PhasesToInterpolationWeights()
{
  this->m_NumberOfReconstructedFrames = 0;
  this->m_UnevenTemporalSpacing = false;
  this->m_SelectedProjections = std::vector<bool>(0);
}

void PhasesToInterpolationWeights::PrintSelf(std::ostream & os, itk::Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << this->m_Array2D << std::endl;
}

void PhasesToInterpolationWeights::SetSelectedProjections(std::vector<bool> sprojs)
{
  this->m_SelectedProjections = sprojs;
  this->Modified();
}

void PhasesToInterpolationWeights::Parse()
{
  itk::SizeValueType rows = 0;
  itk::SizeValueType columns = 0;

  this->PrepareForParsing();

  this->m_InputStream.clear();
  this->m_InputStream.open(this->m_FileName.c_str());
  if ( this->m_InputStream.fail() )
    {
    itkExceptionMacro("The file " << this->m_FileName <<" cannot be opened for reading!"
                                  << std::endl
                                  << "Reason: "
                                  << itksys::SystemTools::GetLastSystemError() );
    }

  // Get the data dimension and set the matrix size
  this->GetDataDimension(rows,columns);
  unsigned int NumberOfProjections = 0;

  // If m_SelectedProjections has not been set, use all projections
  if (m_SelectedProjections.size() == 0)
    NumberOfProjections = rows+1;
  else
    {
    // Check that the size of m_SelectedProjections is consistent with the number of rows in the phase file
    if (!(m_SelectedProjections.size() == rows + 1))
      {
      itkExceptionMacro( "The lists of selected projections and phases have inconsistent sizes" );
      }
    else
      {
      for (unsigned int i=0; i<m_SelectedProjections.size(); i++)
        if (m_SelectedProjections[i]) NumberOfProjections += 1;
      }
    }
  this->m_Array2D.SetSize(this->m_NumberOfReconstructedFrames, NumberOfProjections);

  // Create a vector to hold the projections' phases
  std::vector<float> projectionPhases; //Stores the instant of the cycle at which each projection was acquired
  std::vector<float> reconstructedFrames; //Stores the instant of the cycle each frame represents

  std::string entry;

  // parse the numeric data into the Array2D object
  for (unsigned int j = 0; j < rows+1; j++)
    {
    this->GetNextField(entry);
    if ((m_SelectedProjections.size() == 0) || (m_SelectedProjections[j]))
      projectionPhases.push_back(itk::Math::Round<float>(atof(entry.c_str()) * 1000. ) / 1000.);
    }

  // Compute the instant of the cycle each phase represents
  // Create one more phase point than is strictly necessary, so that projections phases
  // are always between two reconstructed phases. The first and last reconstructed phases
  // represent the same instant. This is accounted for when computing the weights

  if (m_UnevenTemporalSpacing) // Create an unbalanced distribution of phase points : more of them will be placed during systole than during diastole
    {
    std::vector<float> cumulatedWeights;
    cumulatedWeights.push_back(1);
    for (int i=1; i<100; i++)
      {
      float weight = 1;
      if ((i>30) && (i<50)) weight = 2; // The higher this number, the better the temporal resolution in systole
      cumulatedWeights.push_back(cumulatedWeights[i-1] + weight);
      }
    float step = cumulatedWeights[99]/m_NumberOfReconstructedFrames;

    reconstructedFrames.push_back(0);
    for (int n = 1; n < this->m_NumberOfReconstructedFrames; n++)
      {
      int i=0;
      while (cumulatedWeights[reconstructedFrames[n-1] * 100] + step > cumulatedWeights[i])
        {
        i=i+1;
        }
      reconstructedFrames.push_back(((float) i) / 100. );
      }
    reconstructedFrames.push_back(1);
    for (int n = 0; n < this->m_NumberOfReconstructedFrames; n++)
      {
      std::cout << reconstructedFrames[n] << std::endl;
      }
    }
  else   // The reconstructed phases are all separated by the same amount of time
    {
    for (float n = 0.; n < this->m_NumberOfReconstructedFrames + 1; n++)
      {
      reconstructedFrames.push_back(n/this->m_NumberOfReconstructedFrames);
      }
    }
  m_Array2D.Fill(0);

  // Compute the interpolation weights and fill the Array2D
  for (unsigned int c = 0; c < NumberOfProjections; c++)
    {
    int lower = 0; int upper = 1;
    while(!((projectionPhases[c] >= reconstructedFrames[lower]) && (projectionPhases[c] < reconstructedFrames[upper])))
      {
      lower++;
      upper++;
      if(lower == this->m_NumberOfReconstructedFrames)
        {
        std::cout << "Problem while determining the interpolation weights" << std::endl;
        }
      }
    float lowerWeight = (reconstructedFrames[upper] - projectionPhases[c]) / (reconstructedFrames[upper] - reconstructedFrames[lower]);
    float upperWeight = (projectionPhases[c] - reconstructedFrames[lower]) / (reconstructedFrames[upper] - reconstructedFrames[lower]);

    // The last phase is equal to the first one (see comment above when filling "reconstructedFrames")
    if (upper==this->m_NumberOfReconstructedFrames) upper=0;

    m_Array2D[lower][c] = itk::Math::Round<float>(lowerWeight * 100.) / 100.;
    m_Array2D[upper][c] = itk::Math::Round<float>(upperWeight * 100.) / 100.;
    }
  this->m_InputStream.close();
}

/** Update method */

void PhasesToInterpolationWeights::Update()
{
  this->Parse();
}

/** Get the output */

PhasesToInterpolationWeights::Array2DType
PhasesToInterpolationWeights
::GetOutput()
{
  return this->m_Array2D;
}


} //end namespace rtk

#endif
