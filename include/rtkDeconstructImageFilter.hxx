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

#ifndef rtkDeconstructImageFilter_hxx
#define rtkDeconstructImageFilter_hxx

#include "rtkDeconstructImageFilter.h"

namespace rtk
{

/////////////////////////////////////////////////////////
//Default Constructor
template <class TImage>
DeconstructImageFilter<TImage>::DeconstructImageFilter():
  m_NumberOfLevels(5),
  m_Order(3),
  m_PipelineConstructed(false)
{
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
  this->Modified();

  //Set required number of outputs
  int requiredNumberOfOutputs = this->CalculateNumberOfOutputs();
  this->SetNumberOfRequiredOutputs(requiredNumberOfOutputs);

  //Make actual outputs and required outputs match
  int actualNumberOfOutputs = this->GetNumberOfOutputs();

  if (actualNumberOfOutputs < requiredNumberOfOutputs)
    {
    //Add extra outputs
    for (int idx = actualNumberOfOutputs; idx < requiredNumberOfOutputs; idx++)
      {
      typename itk::DataObject::Pointer output = this->MakeOutput(idx);
      this->SetNthOutput(idx, output.GetPointer());
      }
    }
  else if (actualNumberOfOutputs > requiredNumberOfOutputs)
  {
  //Remove extra outputs
  for (int idx = (actualNumberOfOutputs-1); idx >= requiredNumberOfOutputs; idx--)
    {
    if (idx < 0){break;}
    this->RemoveOutput(idx);
    }
  }
}

template <class TImage>
unsigned int DeconstructImageFilter<TImage>::CalculateNumberOfOutputs()
{
  int dimension = TImage::ImageDimension;
  unsigned int n = itk::Math::Round<double>(std::pow(2.0, dimension));
  return (m_NumberOfLevels * (n-1) +1);
}

template <class TImage>
void DeconstructImageFilter<TImage>
::GeneratePassVectors()
{
  int dimension = TImage::ImageDimension;
  unsigned int n = itk::Math::Round<double>(std::pow(2.0, dimension));

  // Create a vector of PassVector
  m_PassVectors.clear();
  for (unsigned int vectIndex = 0; vectIndex < n; vectIndex++)
    {
    typename ConvolutionFilterType::PassVector temp;
    m_PassVectors.push_back(temp);
    }

  // Fill it with the right values
  unsigned int powerOfTwo = 1;
  for (int dim = 0; dim<dimension; dim++)
    {
    for (unsigned int vectIndex = 0; vectIndex < n; vectIndex++)
      {
      // vectIndex / powerOfTwo is a division between unsigned ints, and will return the quotient
      // of their euclidian division
      if ((vectIndex / powerOfTwo)%2) m_PassVectors[vectIndex][dim] = ConvolutionFilterType::High;
      else m_PassVectors[vectIndex][dim] = ConvolutionFilterType::Low;
      }
    powerOfTwo *= 2;
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
  int dimension = TImage::ImageDimension;
  unsigned int n = itk::Math::Round<double>(std::pow(2.0, dimension));

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
  this->GeneratePassVectors();

  // Create all FFTConvolution and Downsampling filters
  for (unsigned int i=0; i<n * m_NumberOfLevels; i++)
    {
    m_ConvolutionFilters.push_back(ConvolutionFilterType::New());
    m_DownsampleFilters.push_back(DownsampleImageFilterType::New());
    }

  // Cascade pipeline
  // Set all the filters and connect them together
  unsigned int *downsamplingFactors = new unsigned int[dimension];
  for (int d=0; d<dimension; d++) downsamplingFactors[d]=2;

  for (unsigned int l=0; l<m_NumberOfLevels; l++)
    {
    for (unsigned int band=0; band<n; band++)
      {
      m_ConvolutionFilters[band + l*n]->SetInput(m_PadFilters[l]->GetOutput());
      m_ConvolutionFilters[band + l*n]->SetPass(m_PassVectors[band]);
      m_ConvolutionFilters[band + l*n]->SetDeconstruction();
      m_ConvolutionFilters[band + l*n]->SetOrder(this->GetOrder());
      m_ConvolutionFilters[band + l*n]->ReleaseDataFlagOn();

      m_DownsampleFilters[band + l*n]->SetInput(m_ConvolutionFilters[band + l*n]->GetOutput());
      m_DownsampleFilters[band + l*n]->SetFactors(downsamplingFactors);

      if ((band > 0) || (l==0))
        {
        m_DownsampleFilters[band + l*n]->ReleaseDataFlagOn();
        }
      }
    if (l<m_NumberOfLevels-1)
      {
      m_PadFilters[l]->SetInput(m_DownsampleFilters[n*(l+1)]->GetOutput());
      }
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
//  std::cout << "Starting deconstruction" << std::endl;

  int dimension = TImage::ImageDimension;
  unsigned int n = itk::Math::Round<double>(std::pow(2.0, dimension));

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
      outputBand++;
      }
    }
//  std::cout << "Done deconstruction" << std::endl;
}


}// end namespace rtk

#endif
