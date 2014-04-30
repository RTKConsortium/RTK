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

#ifndef __rtkReconstructImageFilter_TXX
#define __rtkReconstructImageFilter_TXX

//Includes
#include "rtkReconstructImageFilter.h"

namespace rtk
{

/////////////////////////////////////////////////////////
//Default Constructor
template <class TImage>
ReconstructImageFilter<TImage>::ReconstructImageFilter()
{
  //Initialise private variables
  this->m_NumberOfLevels     = 0;
  this->m_PipelineConstructed = false;
}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void ReconstructImageFilter<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}


/////////////////////////////////////////////////////////
//ModifyInputOutputStorage()
template <class TImage>
void ReconstructImageFilter<TImage>
::ModifyInputOutputStorage()
{
  //Set as modified
  this->Modified();

  //Set required number of outputs
  unsigned int requiredNumberOfInputs = this->CalculateNumberOfInputs();
  this->SetNumberOfRequiredInputs(requiredNumberOfInputs);
}

template <class TImage>
unsigned int ReconstructImageFilter<TImage>::CalculateNumberOfInputs()
{
  unsigned int dimension = TImage::ImageDimension;
  unsigned int n = round(pow(2.0, dimension));
  return (m_NumberOfLevels * (n-1) +1);
}

template <class TImage>
void ReconstructImageFilter<TImage>
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
    m_KernelSources[k]->SetReconstruction();
    m_KernelSources[k]->SetOrder(this->GetOrder());
    }
}

template <class TImage>
void ReconstructImageFilter<TImage>
::GenerateInputRequestedRegion()
{
  for (unsigned int i=0; i<this->CalculateNumberOfInputs(); i++)
    {
    InputImagePointer  inputPtr  = const_cast<TImage *>(this->GetInput(i));
    inputPtr->SetRequestedRegionToLargestPossibleRegion();
    }
}

template <class TImage>
void ReconstructImageFilter<TImage>
::GenerateOutputInformation()
{

  if(!m_PipelineConstructed)
    {
    // n is the number of bands per level, including the ones
    // that will be Reconstructed and won't appear in the outputs
    unsigned int dimension = TImage::ImageDimension;
    unsigned int n = round(pow(2.0, dimension));

    // Before the cascade pipeline
    // Create and set the add filters
    for (unsigned int l=0; l<m_NumberOfLevels; l++)
      {
      m_AddFilters.push_back(AddFilterType::New());
      }

    // Create and set the kernel sources
    this->GenerateAllKernelSources();

    // Create all FFTConvolution and Downsampling filters
    for (unsigned int i=0; i<n * m_NumberOfLevels; i++)
      {
      m_ConvolutionFilters.push_back(ConvolutionFilterType::New());
      m_UpsampleFilters.push_back(UpsampleImageFilterType::New());
      }

    // Cascade pipeline
    // Set all the filters and connect them together
    unsigned int *upsamplingFactors = new unsigned int[dimension];
    for (unsigned int d=0; d<dimension; d++) upsamplingFactors[d]=2;

    for (unsigned int l=0; l<m_NumberOfLevels; l++)
      {
      for (unsigned int band=0; band<n; band++)
        {
        m_ConvolutionFilters[band + l*n]->SetInput(m_UpsampleFilters[band + l*n]->GetOutput());
        m_ConvolutionFilters[band + l*n]->SetKernelImage(m_KernelSources[band]->GetOutput());
        m_ConvolutionFilters[band + l*n]->SetOutputRegionModeToValid();

        m_AddFilters[l]->SetInput(band, m_ConvolutionFilters[band + l*n]->GetOutput());
        m_UpsampleFilters[band + l*n]->SetFactors(upsamplingFactors);
        m_UpsampleFilters[band + l*n]->SetOrder(this->m_Order);
        m_UpsampleFilters[band + l*n]->SetOutputSize(this->m_Sizes[band + l*n]);
        m_UpsampleFilters[band + l*n]->SetOutputIndex(this->m_Indices[band + l*n]);
        }
      if (l>0) m_UpsampleFilters[n*l]->SetInput(m_AddFilters[l-1]->GetOutput());
      }

    // Connect the upsample filters to the inputs of the pipeline
    unsigned int inputBand = 0;
    for (unsigned int i=0; i<n*m_NumberOfLevels; i++)
      {
      if ((i%n) || (i==0))
        {
        m_UpsampleFilters[i]->SetInput(this->GetInput(inputBand));
        inputBand++;
        }
      }

    //Clean up
    delete[] upsamplingFactors;
    }
  m_PipelineConstructed = true;

  // Have the last filter calculate its output information
  // and copy it as the output information of the composite filter
  m_AddFilters[m_NumberOfLevels-1]->UpdateOutputInformation();
  this->GetOutput()->CopyInformation( m_AddFilters[m_NumberOfLevels-1]->GetOutput() );
}

template <class TImage>
void ReconstructImageFilter<TImage>
::GenerateData()
{
  // Have the last filter calculate its output image
  // and graft it to the output of the composite filter
  m_AddFilters[m_NumberOfLevels-1]->Update();
  this->GraftOutput(m_AddFilters[m_NumberOfLevels-1]->GetOutput() );
}


}// end namespace rtk

#endif
