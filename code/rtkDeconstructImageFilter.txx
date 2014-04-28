/*=========================================================================

  Program:  rtk Multi-level Multi-band Image Filter
  Module:   rtkDeconstructImageFilter.txx
  Language: C++
  Date:     2005/11/22
  Version:  0.2
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See rtkCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
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
//    m_KernelSources[k]->Update();
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
    if (l>0) m_PadFilters[l]->SetInput(m_DownsampleFilters[n*(l-1)]->GetOutput());
    }
  m_PadFilters[0]->SetInput(this->GetInput());

  // Have the last filters calculate their output information
  // and copy it as the output information of the composite filter
  // Since not all downsample filters are connected to an output,
  // a conversion between their indices and the indices of the outputs is required
  unsigned int outputBand = this->CalculateNumberOfOutputs()-1;
  for (unsigned int l=0; l<m_NumberOfLevels; l++)
    {
    for (int band=n-1; band>=0; band--)
      {
      if (band>0 || l==(m_NumberOfLevels-1))
        {
        m_DownsampleFilters[band + l*n]->UpdateOutputInformation();
        this->GetOutput(outputBand)->CopyInformation( m_DownsampleFilters[band + l*n]->GetOutput() );
//        this->GetOutput(outputBand)->Print(std::cout);
        outputBand--;
        }
      }
    }

  //Clean up
  delete[] downsamplingFactors;
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
  // a conversion between their indices and the indices of the outputs is required
  unsigned int outputBand = this->CalculateNumberOfOutputs()-1;
  for (unsigned int l=0; l<m_NumberOfLevels; l++)
    {
    for (int band=n-1; band>=0; band--)
      {
      if (band || l==(m_NumberOfLevels-1))
        {
        m_DownsampleFilters[band + l*n]->GraftOutput(this->GetOutput(outputBand));
        m_DownsampleFilters[band + l*n]->Update();
        this->GraftNthOutput(outputBand, m_DownsampleFilters[band + l*n]->GetOutput() );
        outputBand--;
        }
      }
    }
}


}// end namespace rtk

#endif
