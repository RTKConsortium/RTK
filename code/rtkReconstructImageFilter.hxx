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

#ifndef rtkReconstructImageFilter_hxx
#define rtkReconstructImageFilter_hxx

//Includes
#include "rtkReconstructImageFilter.h"

namespace rtk
{

/////////////////////////////////////////////////////////
//Default Constructor
template <class TImage>
ReconstructImageFilter<TImage>::ReconstructImageFilter():
  m_NumberOfLevels(5),
  m_Order(3),
  m_PipelineConstructed(false)
{
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
  this->Modified();

  //Set required number of inputs
  unsigned int requiredNumberOfInputs = this->CalculateNumberOfInputs();
  this->SetNumberOfRequiredInputs(requiredNumberOfInputs);
}

template <class TImage>
unsigned int ReconstructImageFilter<TImage>::CalculateNumberOfInputs()
{
  int dimension = TImage::ImageDimension;
  unsigned int n = itk::Math::Round<double>(std::pow(2.0, dimension));
  return (m_NumberOfLevels * (n-1) +1);
}

template <class TImage>
void ReconstructImageFilter<TImage>
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
    int dimension = TImage::ImageDimension;
    unsigned int n = itk::Math::Round<double>(std::pow(2.0, dimension));

    // Before the cascade pipeline
    // Create and set the add filters
    for (unsigned int l=0; l<m_NumberOfLevels; l++)
      {
      m_AddFilters.push_back(AddFilterType::New());
      }

    // Create and set the kernel sources
    this->GeneratePassVectors();

    // Create all FFTConvolution and Downsampling filters
    for (unsigned int i=0; i<n * m_NumberOfLevels; i++)
      {
      m_ConvolutionFilters.push_back(ConvolutionFilterType::New());
      m_UpsampleFilters.push_back(UpsampleImageFilterType::New());
      }

    // Cascade pipeline
    // Set all the filters and connect them together
    unsigned int *upsamplingFactors = new unsigned int[dimension];
    for (int d=0; d<dimension; d++) upsamplingFactors[d]=2;

    for (unsigned int l=0; l<m_NumberOfLevels; l++)
      {
      for (unsigned int band=0; band<n; band++)
        {
        m_ConvolutionFilters[band + l*n]->SetInput(m_UpsampleFilters[band + l*n]->GetOutput());
        m_ConvolutionFilters[band + l*n]->SetPass(m_PassVectors[band]);
        m_ConvolutionFilters[band + l*n]->SetReconstruction();
        m_ConvolutionFilters[band + l*n]->SetOrder(this->GetOrder());
        m_ConvolutionFilters[band + l*n]->ReleaseDataFlagOn();

        m_AddFilters[l]->SetInput(band, m_ConvolutionFilters[band + l*n]->GetOutput());
        m_AddFilters[l]->ReleaseDataFlagOn();

        m_UpsampleFilters[band + l*n]->SetFactors(upsamplingFactors);
        m_UpsampleFilters[band + l*n]->SetOrder(this->m_Order);
        m_UpsampleFilters[band + l*n]->SetOutputSize(this->m_Sizes[band + l*n]);
        m_UpsampleFilters[band + l*n]->SetOutputIndex(this->m_Indices[band + l*n]);
        m_UpsampleFilters[band + l*n]->ReleaseDataFlagOn();
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
//  std::cout << "Starting reconstruction" << std::endl;

  int dimension = TImage::ImageDimension;
  unsigned int n = itk::Math::Round<double>(std::pow(2.0, dimension));

  // Have the last filter calculate its output image
  // and graft it to the output of the composite filter
  for (unsigned int l=0; l<m_NumberOfLevels; l++)
    {
      for (unsigned int band=0; band<n; band++)
        {
        m_UpsampleFilters[band + l*n]->Update();
        m_ConvolutionFilters[band + l*n]->Update();
        }
    m_AddFilters[l]->Update();
    }
  this->GraftOutput(m_AddFilters[m_NumberOfLevels-1]->GetOutput() );

//  std::cout << "Done reconstruction" << std::endl;
}


}// end namespace rtk

#endif
