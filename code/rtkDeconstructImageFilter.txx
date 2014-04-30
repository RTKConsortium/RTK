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

#ifndef __rtkDeconstructImageFilter_TXX
#define __rtkDeconstructImageFilter_TXX

//Includes
#include "rtkDeconstructImageFilter.h"

namespace rtk
{

/////////////////////////////////////////////////////////
//Default Constructor
template <class TImage>
DeconstructImageFilter<TImage>::DeconstructImageFilter()
{
  //Initialise private variables
  this->m_NumberOfLevels     = 0;
  this->m_PipelineConstructed = false;
}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void DeconstructImageFilter<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}


/////////////////////////////////////////////////////////
//ModifyInputOutputStorage()
template <class TImage>
void DeconstructImageFilter<TImage>
::ModifyInputOutputStorage()
{
    //Set as modified
    this->Modified();

    //Set required number of outputs
    unsigned int requiredNumberOfOutputs = this->CalculateNumberOfOutputs();
    this->SetNumberOfRequiredOutputs(requiredNumberOfOutputs);

    //Make actual outputs and required outputs match
    int actualNumberOfOutputs = this->GetNumberOfOutputs();
    int idx;

    if (actualNumberOfOutputs < requiredNumberOfOutputs)
    {
        //Add extra outputs
        for (idx = actualNumberOfOutputs; idx < requiredNumberOfOutputs; idx++)
        {
            typename itk::DataObject::Pointer output = this->MakeOutput(idx);
            this->SetNthOutput(idx, output.GetPointer());
        }
    }
    else if (actualNumberOfOutputs > requiredNumberOfOutputs)
    {
        //Remove extra outputs
        for (idx = (actualNumberOfOutputs-1); idx >= requiredNumberOfOutputs; idx--)
        {
            if (idx < 0){break;}
            this->RemoveOutput(idx);
        }
    }
}

template <class TImage>
unsigned int DeconstructImageFilter<TImage>::CalculateNumberOfOutputs()
{
  unsigned int dimension = TImage::ImageDimension;
  unsigned int n = round(pow(2.0, dimension));
  return (m_NumberOfLevels * (n-1) +1);
}

template <class TImage>
void DeconstructImageFilter<TImage>
::GenerateAllKernelSources()
{
  unsigned int dimension = TImage::ImageDimension;
  unsigned int n = round(pow(2.0, dimension));

  // Create a vector of PassVector
  typename KernelSourceType::PassVector *passVectors = new typename KernelSourceType::PassVector[n];

  // Fill it with the right values
  unsigned int powerOfTwo = 1;
  for (unsigned int dim = 0; dim<dimension; dim++)
    {
    for (unsigned int vectIndex = 0; vectIndex < n; vectIndex++)
      {
      if ((int)floor(vectIndex / powerOfTwo)%2) passVectors[vectIndex][dim] = KernelSourceType::High;
      else passVectors[vectIndex][dim] = KernelSourceType::Low;
      }
    powerOfTwo *= 2;
    }

  // Generate all the kernel sources and store them into m_KernelSources
  for (unsigned int k=0; k<n; k++)
    {
    m_KernelSources.push_back(KernelSourceType::New());
    m_KernelSources[k]->SetPass(passVectors[k]);
    m_KernelSources[k]->SetDeconstruction();
    m_KernelSources[k]->SetOrder(this->GetOrder());
    }
}

template <class TImage>
void DeconstructImageFilter<TImage>
::GenerateInputRequestedRegion()
{
  InputImagePointer  inputPtr  = const_cast<TImage *>(this->GetInput());
  inputPtr->SetRequestedRegionToLargestPossibleRegion();
}

template <class TImage>
void DeconstructImageFilter<TImage>
::GenerateOutputInformation()
{

  // n is the number of bands per level, including the ones
  // that will be deconstructed and won't appear in the outputs
  unsigned int dimension = TImage::ImageDimension;
  unsigned int n = round(pow(2.0, dimension));

  if(!m_PipelineConstructed)
    {

  // Before the cascade pipeline
  // Create and set the padding filters
  for (unsigned int l=0; l<m_NumberOfLevels; l++)
    {
    m_PadFilters.push_back(PadFilterType::New());
    typename TImage::SizeType padSize;
    padSize.Fill(2 * m_Order - 1);
    m_PadFilters[l]->SetPadBound(padSize);
    }

  // Create and set the kernel sources
  this->GenerateAllKernelSources();

  // Create all FFTConvolution and Downsampling filters
  for (unsigned int i=0; i<n * m_NumberOfLevels; i++)
    {
    m_ConvolutionFilters.push_back(ConvolutionFilterType::New());
    m_DownsampleFilters.push_back(DownsampleImageFilterType::New());
    }

  // Cascade pipeline
  // Set all the filters and connect them together
  unsigned int *downsamplingFactors = new unsigned int[dimension];
  for (unsigned int d=0; d<dimension; d++) downsamplingFactors[d]=2;

  for (unsigned int l=0; l<m_NumberOfLevels; l++)
    {
    for (unsigned int band=0; band<n; band++)
      {
      m_ConvolutionFilters[band + l*n]->SetInput(m_PadFilters[l]->GetOutput());
      m_ConvolutionFilters[band + l*n]->SetKernelImage(m_KernelSources[band]->GetOutput());
      m_ConvolutionFilters[band + l*n]->SetOutputRegionModeToValid();

      m_DownsampleFilters[band + l*n]->SetInput(m_ConvolutionFilters[band + l*n]->GetOutput());
      m_DownsampleFilters[band + l*n]->SetFactors(downsamplingFactors);
      }
    if (l<m_NumberOfLevels-1) m_PadFilters[l]->SetInput(m_DownsampleFilters[n*(l+1)]->GetOutput());
    }
  m_PadFilters[m_NumberOfLevels-1]->SetInput(this->GetInput());

  //Clean up
  delete[] downsamplingFactors;

    }

  // Have the last filters calculate their output information
  // and copy it as the output information of the composite filter
  // Since not all downsample filters are connected to an output,
  // some are skipped
  unsigned int outputBand = 0;
  for (unsigned int i=0; i<n*m_NumberOfLevels; i++)
    {
    if ((i%n) || (i==0))
      {
      m_DownsampleFilters[i]->UpdateOutputInformation();
      this->GetOutput(outputBand)->CopyInformation( m_DownsampleFilters[i]->GetOutput() );
//      this->GetOutput(outputBand)->Print(std::cout);
      outputBand++;
      }
    }

  if(!m_PipelineConstructed)
    {
    m_Sizes.clear();
    m_Indices.clear();
    for (unsigned int i=0; i<n*m_NumberOfLevels; i++)
      {
      m_Sizes.push_back(m_ConvolutionFilters[i]->GetOutput()->GetLargestPossibleRegion().GetSize());
      m_Indices.push_back(m_ConvolutionFilters[i]->GetOutput()->GetLargestPossibleRegion().GetIndex());
      }
    }

  m_PipelineConstructed = true;
}

template <class TImage>
void DeconstructImageFilter<TImage>
::GenerateData()
{
  unsigned int dimension = TImage::ImageDimension;
  unsigned int n = round(pow(2.0, dimension));

  // Have the last filters calculate their output image
  // and graft it to the output of the composite filter
  // Since not all downsample filters are connected to an output,
  // some are skipped
  unsigned int outputBand = 0;
  for (unsigned int i=0; i<n*m_NumberOfLevels; i++)
    {
    if ((i%n) || (i==0))
      {
      m_DownsampleFilters[i]->Update();
      this->GraftNthOutput(outputBand, m_DownsampleFilters[i]->GetOutput() );
//      std::cout << "Grafting " << i << "-th downsample filter's output to output " << outputBand << std::endl;
      outputBand++;
      }
    }
  // Debugging
//  std::cout << "************ Printing output of Downsample filter 3 *************" << std::endl;
//  m_DownsampleFilters[3]->GetOutput()->Print(std::cout);
//  std::cout << "************ Printing output of Downsample filter 4 *************" << std::endl;
//  m_DownsampleFilters[4]->GetOutput()->Print(std::cout);
//  std::cout << "************ Printing input of Downsample filter 4 *************" << std::endl;
//  m_DownsampleFilters[4]->GetInput()->Print(std::cout);
//  std::cout << "************ Printing output of Pad filter 1 *************" << std::endl;
//  m_PadFilters[1]->GetOutput()->Print(std::cout);
//  std::cout << "************ Printing output of convolution filter 4 *************" << std::endl;
//  m_ConvolutionFilters[1]->GetOutput()->Print(std::cout);
//  std::cout << "************ Printing input of convolution filter 4 *************" << std::endl;
//  m_ConvolutionFilters[1]->GetInput()->Print(std::cout);

}


}// end namespace rtk

#endif
