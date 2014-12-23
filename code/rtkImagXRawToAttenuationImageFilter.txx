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
  m_CropFilter = CropFilterType::New();
  m_ExtractFilter = ExtractFilterType::New();
  m_BinningFilter = BinningFilterType::New();
  m_ScatterFilter = ScatterFilterType::New();
  m_I0estimationFilter = I0FilterType::New();
  m_LookupTableFilter = LookupTableFilterType::New();
  m_PasteFilter = PasteFilterType::New();
  m_ConstantSource = ConstantImageSourceType::New();

  //Permanent internal connections
  m_ExtractFilter->SetInput(m_CropFilter->GetOutput());

  m_BinningFilter->SetInput(m_ExtractFilter->GetOutput());
  m_ScatterFilter->SetInput(m_BinningFilter->GetOutput());
  m_I0estimationFilter->SetInput(m_ScatterFilter->GetOutput());
  m_LookupTableFilter->SetInput( m_I0estimationFilter->GetOutput() );

  // Set permanent connections
  m_PasteFilter->SetDestinationImage(m_ConstantSource->GetOutput());
  m_PasteFilter->SetSourceImage(m_LookupTableFilter->GetOutput());

  //Default filter parameters
  typename CropFilterType::SizeType border = m_CropFilter->GetLowerBoundaryCropSize();
  border[0] = 4;
  border[1] = 4;
  m_CropFilter->SetBoundaryCropSize(border);

  m_BinningFilter->SetShrinkFactor(0, 2);
  m_BinningFilter->SetShrinkFactor(1, 2);
  m_BinningFilter->SetShrinkFactor(2, 1);
}

template<class TOutputImage, unsigned char bitShift>
void
ImagXRawToAttenuationImageFilter<TOutputImage, bitShift>
::GenerateOutputInformation()
{
  int Dimension = InputImageType::ImageDimension;

  m_CropFilter->SetInput(this->GetInput());
  m_CropFilter->UpdateOutputInformation();

  // Set extraction regions and indices
  m_ExtractRegion = m_CropFilter->GetOutput()->GetLargestPossibleRegion();
  m_ExtractRegion.SetSize(Dimension - 1, 1);
  m_ExtractRegion.SetIndex(Dimension - 1, 0);

  m_ExtractFilter->SetExtractionRegion(m_ExtractRegion);
  m_ExtractFilter->UpdateOutputInformation();
  
  // Initialize the source
  m_LookupTableFilter->UpdateOutputInformation();
  m_ConstantSource->SetInformationFromImage(m_LookupTableFilter->GetOutput());
  typename OutputImageType::SizeType size = m_LookupTableFilter->GetOutput()->GetLargestPossibleRegion().GetSize();
  size[OutputImageType::ImageDimension - 1] = m_CropFilter->GetOutput()->GetLargestPossibleRegion().GetSize(OutputImageType::ImageDimension-1);
  m_ConstantSource->SetSize(size);

  // Set extraction regions and indices
  m_PasteRegion = m_LookupTableFilter->GetOutput()->GetLargestPossibleRegion();
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
  
  m_CropFilter->Update();

  for (int frame = 0; frame < this->GetInput()->GetLargestPossibleRegion().GetSize(Dimension - 1); frame++)
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
