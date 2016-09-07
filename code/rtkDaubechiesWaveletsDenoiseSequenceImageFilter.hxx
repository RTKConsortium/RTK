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

#ifndef rtkDaubechiesWaveletsDenoiseSequenceImageFilter_hxx
#define rtkDaubechiesWaveletsDenoiseSequenceImageFilter_hxx

#include "rtkDaubechiesWaveletsDenoiseSequenceImageFilter.h"
#include <itkImageFileWriter.h>

namespace rtk
{

template< typename TImageSequence>
DaubechiesWaveletsDenoiseSequenceImageFilter< TImageSequence>
::DaubechiesWaveletsDenoiseSequenceImageFilter():
  m_Order(5),
  m_Threshold(1),
  m_NumberOfLevels(3)
{
  // Create the filters
  m_WaveletsDenoisingFilter = WaveletsDenoisingFilterType::New();
  m_ExtractFilter = ExtractFilterType::New();
  m_PasteFilter = PasteFilterType::New();
  m_CastFilter = CastFilterType::New();
  m_ConstantSource = ConstantImageSourceType::New();

  // Set permanent connections
  m_WaveletsDenoisingFilter->SetInput(m_ExtractFilter->GetOutput());
  m_CastFilter->SetInput(m_WaveletsDenoisingFilter->GetOutput());
  m_PasteFilter->SetSourceImage(m_CastFilter->GetOutput());

  // Set permanent parameters
  m_ExtractFilter->SetDirectionCollapseToIdentity();

  // Set memory management parameters
  m_CastFilter->SetInPlace(false);
}

template< typename TImageSequence>
void
DaubechiesWaveletsDenoiseSequenceImageFilter< TImageSequence>
::GenerateOutputInformation()
{
  int Dimension = TImageSequence::ImageDimension;

  // Set runtime connections
  m_ExtractFilter->SetInput(this->GetInput());

  // Initialize the source
  m_ConstantSource->SetInformationFromImage(this->GetInput());
  m_ConstantSource->Update();
  m_PasteFilter->SetDestinationImage(m_ConstantSource->GetOutput());

  // Set runtime parameters
  m_WaveletsDenoisingFilter->SetOrder(m_Order);
  m_WaveletsDenoisingFilter->SetThreshold(m_Threshold);
  m_WaveletsDenoisingFilter->SetNumberOfLevels(m_NumberOfLevels);

  // Set extraction regions and indices
  m_ExtractAndPasteRegion = this->GetInput()->GetLargestPossibleRegion();
  m_ExtractAndPasteRegion.SetSize(Dimension - 1, 0);
  m_ExtractAndPasteRegion.SetIndex(Dimension - 1, 0);

  m_ExtractFilter->SetExtractionRegion(m_ExtractAndPasteRegion);
  m_ExtractFilter->UpdateOutputInformation();
  m_CastFilter->UpdateOutputInformation();

  m_PasteFilter->SetSourceRegion(m_CastFilter->GetOutput()->GetLargestPossibleRegion());
  m_PasteFilter->SetDestinationIndex(m_ExtractAndPasteRegion.GetIndex());

  // Have the last filter calculate its output information
  m_PasteFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_PasteFilter->GetOutput() );
}


template< typename TImageSequence>
void
DaubechiesWaveletsDenoiseSequenceImageFilter< TImageSequence>
::GenerateInputRequestedRegion()
{
  //Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  //Get pointers to the input and output
  typename TImageSequence::Pointer  inputPtr  = const_cast<TImageSequence *>(this->GetInput());
  inputPtr->SetRequestedRegionToLargestPossibleRegion();
}

template< typename TImageSequence>
void
DaubechiesWaveletsDenoiseSequenceImageFilter< TImageSequence>
::GenerateData()
{
  int Dimension = TImageSequence::ImageDimension;

  // Declare an image pointer to disconnect the output of paste
  typename TImageSequence::Pointer pimg;

  for (unsigned int frame=0; frame<this->GetInput(0)->GetLargestPossibleRegion().GetSize(Dimension-1); frame++)
    {
    if (frame > 0) // After the first frame, use the output of paste as input
      {
      pimg = m_PasteFilter->GetOutput();
      pimg->DisconnectPipeline();
      m_PasteFilter->SetDestinationImage(pimg);
      }

    m_ExtractAndPasteRegion.SetIndex(Dimension - 1, frame);

    m_ExtractFilter->SetExtractionRegion(m_ExtractAndPasteRegion);
    m_ExtractFilter->UpdateLargestPossibleRegion();

    m_WaveletsDenoisingFilter->Update();

    m_CastFilter->Update();

    m_PasteFilter->SetSourceRegion(m_CastFilter->GetOutput()->GetLargestPossibleRegion());
    m_PasteFilter->SetDestinationIndex(m_ExtractAndPasteRegion.GetIndex());

    m_PasteFilter->UpdateLargestPossibleRegion();
    }
  this->GraftOutput( m_PasteFilter->GetOutput() );

  m_ExtractFilter->GetOutput()->ReleaseData();
  m_WaveletsDenoisingFilter->GetOutput()->ReleaseData();
  m_CastFilter->GetOutput()->ReleaseData();
}


}// end namespace


#endif
