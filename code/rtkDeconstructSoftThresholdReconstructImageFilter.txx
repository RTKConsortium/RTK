#ifndef __rtkDeconstructSoftThresholdReconstructImageFilter_TXX
#define __rtkDeconstructSoftThresholdReconstructImageFilter_TXX

//rtk Includes
#include "rtkDeconstructSoftThresholdReconstructImageFilter.h"

namespace rtk
{

/////////////////////////////////////////////////////////
//Constructor()
template <class TImage>
DeconstructSoftThresholdReconstructImageFilter<TImage>
::DeconstructSoftThresholdReconstructImageFilter()
{
    m_DeconstructionFilter = DeconstructFilterType::New();
    m_ReconstructionFilter = ReconstructFilterType::New();
    m_WaveletsOrder = 3;
}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void
DeconstructSoftThresholdReconstructImageFilter<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}

/////////////////////////////////////////////////////////
// Pass the number of decomposition levels to the wavelet
// filters
template <class TImage>
void
DeconstructSoftThresholdReconstructImageFilter<TImage>
::SetNumberOfLevels(unsigned int levels)
{
    m_DeconstructionFilter->SetNumberOfLevels(levels);
    m_ReconstructionFilter->SetNumberOfLevels(levels);
}

/////////////////////////////////////////////////////////
//GenerateData()
template <class TImage>
void
DeconstructSoftThresholdReconstructImageFilter<TImage>
::GenerateData()
{
    // Setup the deconstruction and reconstruction filters
    m_DeconstructionFilter->SetInput(this->GetInput());

    // Perform deconstruction
    m_DeconstructionFilter->Update();

    //Create and setup an array of soft threshold filters
    typename SoftThresholdFilterType::Pointer softThresholdFilterArray[m_DeconstructionFilter->GetNumberOfOutputs()];

    int NbDetailsPerLevel = pow(2.0, (double) ImageDimension);

    //Perform soft thresholding and set inputs for reconstruction
    for (unsigned int index=0; index < m_DeconstructionFilter->GetNumberOfOutputs(); index++)
      {
      if ((index % NbDetailsPerLevel)==0)
        {
        // Do not soft threshold the low pass coefficients
        m_ReconstructionFilter->SetInput(index, m_DeconstructionFilter->GetOutput(index));
        }
      else
        {
        // Soft thresholding
        softThresholdFilterArray[index] = SoftThresholdFilterType::New();
        softThresholdFilterArray[index]->SetInput(m_DeconstructionFilter->GetOutput(index));
        softThresholdFilterArray[index]->SetThreshold(m_Threshold);

        //Set input for reconstruction
        m_ReconstructionFilter->SetInput(index, softThresholdFilterArray[index]->GetOutput());

        } //end if
      }// end for

    // Perform reconstruction
    m_ReconstructionFilter->Update();

    this->GraftOutput( m_ReconstructionFilter->GetOutput() );

}


}// end namespace rtk

#endif
