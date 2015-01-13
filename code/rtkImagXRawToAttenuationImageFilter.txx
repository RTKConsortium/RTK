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

#ifndef __rtkImagXRawToAttenuationImageFilter_txx
#define __rtkImagXRawToAttenuationImageFilter_txx

#include <itkImageFileWriter.h>

namespace rtk
{
  
template<class TOutputImage, unsigned char bitShift>
ImagXRawToAttenuationImageFilter<TOutputImage, bitShift>
::ImagXRawToAttenuationImageFilter()
{
  m_ExtractFilter = ExtractFilterType::New();
  m_CropFilter = CropFilterType::New();
  m_BinningFilter = BinningFilterType::New();
  m_ScatterFilter = ScatterFilterType::New();
  m_I0estimationFilter = I0FilterType::New();
  m_LookupTableFilter = LookupTableFilterType::New();
  m_WpcFilter = WpcType::New();
  m_PasteFilter = PasteFilterType::New();
  m_ConstantSource = ConstantImageSourceType::New();

  //Permanent internal connections
  m_ExtractFilter->SetInput(this->GetInput());
  m_CropFilter->SetInput(m_ExtractFilter->GetOutput());
  m_BinningFilter->SetInput(m_CropFilter->GetOutput());
  m_ScatterFilter->SetInput(m_BinningFilter->GetOutput());
  m_I0estimationFilter->SetInput(m_ScatterFilter->GetOutput());
  m_LookupTableFilter->SetInput( m_I0estimationFilter->GetOutput() );
  m_WpcFilter->SetInput(m_LookupTableFilter->GetOutput());

  // Set permanent connections
  m_PasteFilter->SetDestinationImage(m_ConstantSource->GetOutput());
  m_PasteFilter->SetSourceImage(m_WpcFilter->GetOutput());

  //Default filter parameters
  m_LowerBoundaryCropSize[0] = 0;
  m_LowerBoundaryCropSize[1] = 0;
  m_UpperBoundaryCropSize[0] = 0;
  m_UpperBoundaryCropSize[1] = 0;

  m_BinningKernelSize.push_back(1);
  m_BinningKernelSize.push_back(1);
  m_BinningKernelSize.push_back(1); // Must stay at 1!
  
  m_ScatterToPrimaryRatio = 0.0;
  m_AirThreshold = 10000.;         // BAD!

  m_WpcCoefficients.push_back(0.0);
  m_WpcCoefficients.push_back(1.0);
}

template<class TOutputImage, unsigned char bitShift>
void
ImagXRawToAttenuationImageFilter<TOutputImage, bitShift>
::GenerateOutputInformation()
{
  int Dimension = InputImageType::ImageDimension;
  
  // Set extraction regions and indices
  m_ExtractFilter->SetInput(this->GetInput());
  m_ExtractRegion = this->GetInput()->GetLargestPossibleRegion();
  m_ExtractRegion.SetSize(Dimension - 1, 1);
  m_ExtractRegion.SetIndex(Dimension - 1, 0);
  m_ExtractFilter->SetExtractionRegion(m_ExtractRegion);
  m_ExtractFilter->UpdateOutputInformation();
  
  m_CropFilter->SetInput(m_ExtractFilter->GetOutput());
  m_CropFilter->SetUpperBoundaryCropSize(m_UpperBoundaryCropSize);
  m_CropFilter->SetLowerBoundaryCropSize(m_LowerBoundaryCropSize);
  m_CropFilter->UpdateOutputInformation();

  m_BinningFilter->SetShrinkFactor(0, m_BinningKernelSize[0]);
  m_BinningFilter->SetShrinkFactor(1, m_BinningKernelSize[1]);
  m_BinningFilter->SetShrinkFactor(2, 1);
  
  // Initialize the source
  m_WpcFilter->UpdateOutputInformation();
  m_ConstantSource->SetInformationFromImage(m_WpcFilter->GetOutput());
  typename OutputImageType::SizeType size = m_WpcFilter->GetOutput()->GetLargestPossibleRegion().GetSize();
  size[OutputImageType::ImageDimension - 1] = this->GetInput()->GetLargestPossibleRegion().GetSize(OutputImageType::ImageDimension - 1);
  m_ConstantSource->SetSize(size);

  // Set extraction regions and indices
  m_PasteRegion = m_WpcFilter->GetOutput()->GetLargestPossibleRegion();
  m_PasteRegion.SetSize(Dimension - 1, 1);
  m_PasteRegion.SetIndex(Dimension - 1, 0);
  
  m_ConstantSource->Update();
  m_PasteFilter->SetDestinationImage(m_ConstantSource->GetOutput());
  m_PasteFilter->SetSourceRegion(m_ConstantSource->GetOutput()->GetLargestPossibleRegion());
  m_PasteFilter->SetDestinationIndex(m_PasteRegion.GetIndex());

  // Have the last filter calculate its output information
  m_PasteFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation(m_PasteFilter->GetOutput());
}

template<class TOutputImage, unsigned char bitShift>
void
ImagXRawToAttenuationImageFilter<TOutputImage, bitShift>
::GenerateData()
{
  int Dimension = InputImageType::ImageDimension;

  // Declare an image pointer to disconnect the output of paste
  typename OutputImageType::Pointer pimg;
    
//  m_ScatterFilter->SetAirThreshold(m_AirThreshold);
//  m_ScatterFilter->SetScatterToPrimaryRatio(m_ScatterToPrimaryRatio);
  m_WpcFilter->SetCoefficients(m_WpcCoefficients);

  m_ExtractFilter->Update();

  for (unsigned int frame = 0; frame < this->GetInput()->GetLargestPossibleRegion().GetSize(Dimension - 1); frame++)
    {
    if (frame > 0) // After the first frame, use the output of paste as input
      {
      pimg = m_PasteFilter->GetOutput();
      pimg->DisconnectPipeline();
      m_PasteFilter->SetDestinationImage(pimg);
      }

    m_ExtractRegion.SetIndex(Dimension - 1, frame);
    m_PasteRegion.SetIndex(Dimension - 1, frame);

    m_ExtractFilter->SetExtractionRegion(m_ExtractRegion);
    m_ExtractFilter->UpdateLargestPossibleRegion();
   
    m_I0estimationFilter->UpdateLargestPossibleRegion();
 
    m_LookupTableFilter->SetI0(m_I0estimationFilter->GetI0());
    m_LookupTableFilter->UpdateLargestPossibleRegion();
    
    m_PasteFilter->SetSourceRegion(m_LookupTableFilter->GetOutput()->GetLargestPossibleRegion());
    m_PasteFilter->SetDestinationIndex(m_PasteRegion.GetIndex());

    m_PasteFilter->Update();
    }
  this->GraftOutput(m_PasteFilter->GetOutput());
}

} // end namespace rtk
#endif
