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
#ifndef __rtkPhasesToInterpolationWeights_txx
#define __rtkPhasesToInterpolationWeights_txx

#include "rtkPhasesToInterpolationWeights.h"

#include "itksys/SystemTools.hxx"
#include <vcl_limits.h>

namespace rtk
{

PhasesToInterpolationWeights::PhasesToInterpolationWeights()
{
    //    this->m_Array2D = new Array2DType;
    this->m_NumberReconstructedFrames = 0;
    this->m_UnevenTemporalSpacing = false;
}

void PhasesToInterpolationWeights::PrintSelf(std::ostream & os, itk::Indent indent) const
{
    Superclass::PrintSelf(os,indent);
    os << this->m_Array2D << std::endl;
}

void PhasesToInterpolationWeights::SetNumberOfReconstructedFrames(int n){
    this->m_NumberReconstructedFrames = n;
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
        itkExceptionMacro(
                    "The file " << this->m_FileName <<" cannot be opened for reading!"
                    << std::endl
                    << "Reason: "
                    << itksys::SystemTools::GetLastSystemError() );
    }

    // Get the data dimension and set the matrix size
    this->GetDataDimension(rows,columns);
    unsigned int NumberOfProjections = rows+1;
    this->m_Array2D.SetSize(this->m_NumberReconstructedFrames, NumberOfProjections);

    // Create a vector to hold the projections' phases
    std::vector<float> projectionPhases; //Stores the instant of the cardiac cycle at which each projection was acquired
    std::vector<float> reconstructedPhases; //Stores the instant of the cardiac cycle each phase represents

    std::string entry;

    // parse the numeric data into the Array2D object
    for (unsigned int j = 0; j < NumberOfProjections; j++)
    {
        this->GetNextField(entry);
        projectionPhases.push_back(atof(entry.c_str()));

        //        /** if the file contains missing data, m_Line will contain less data
        //       * fields. So we check if m_Line is empty and if it is, we break out of
        //       * this loop and move to the next line. */
        //        if ( this->m_Line.empty() )
        //        {
        //            break;
        //        }
    }

    // Compute the instant of the cardiac cycle each phase represents
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
        float step = cumulatedWeights[99]/m_NumberReconstructedFrames;

        reconstructedPhases.push_back(0);
        for (int n = 1; n < this->m_NumberReconstructedFrames; n++)
        {
            int i=0;
            while (cumulatedWeights[reconstructedPhases[n-1] * 100] + step > cumulatedWeights[i])
            {
                i=i+1;
            }

            reconstructedPhases.push_back(((float) i) / 100. );
        }
        reconstructedPhases.push_back(1);
        for (int n = 0; n < this->m_NumberReconstructedFrames; n++){
            std::cout << reconstructedPhases[n] << std::endl;
        }
    }
    else   // The reconstructed phases are all separated by the same amount of time
    {
        for (float n = 0.; n < this->m_NumberReconstructedFrames + 1; n++)
        {
            reconstructedPhases.push_back(n/this->m_NumberReconstructedFrames);
        }
    }
    m_Array2D.Fill(0);


    // Compute the interpolation weights and fill the Array2D
    for (unsigned int c = 0; c < NumberOfProjections; c++)
    {
        int lower = 0; int upper = 1;
        while(!((projectionPhases[c] >= reconstructedPhases[lower]) && (projectionPhases[c] < reconstructedPhases[upper]))) {
            lower++;
            upper++;
            if(lower == this->m_NumberReconstructedFrames) {
                std::cout << "Problem while determining the interpolation weights" << std::endl;
            }
        }
        float lowerWeight = (reconstructedPhases[upper] - projectionPhases[c]) / (reconstructedPhases[upper] - reconstructedPhases[lower]);
        float upperWeight = (projectionPhases[c] - reconstructedPhases[lower]) / (reconstructedPhases[upper] - reconstructedPhases[lower]);

        // The last phase is equal to the first one (see comment above when filling "reconstructedPhases")
        if (upper==this->m_NumberReconstructedFrames) upper=0;

        m_Array2D[lower][c] = (float) lowerWeight;
        m_Array2D[upper][c] = (float) upperWeight;

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
