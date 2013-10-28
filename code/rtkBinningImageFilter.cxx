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

#ifndef __rtkBinningImageFilter_cxx
#define __rtkBinningImageFilter_cxx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkVariableLengthVector.h>
#include <itkImageConstIteratorWithIndex.h>


namespace rtk
{

BinningImageFilter::BinningImageFilter()
{
  m_BinningFactors[0]=2;
  m_BinningFactors[1]=2;
}

void BinningImageFilter::GenerateInputRequestedRegion()
{
  ImageType::Pointer inputPtr = const_cast<ImageType *>(this->GetInput());
  const ImageType::Pointer outputPtr = const_cast<ImageType *>(this->GetOutput());

  ImageType::RegionType inputReqRegion = inputPtr->GetLargestPossibleRegion();
  const ImageType::RegionType outputReqRegion = outputPtr->GetRequestedRegion();
  const ImageType::RegionType outputLPRegion = outputPtr->GetLargestPossibleRegion();
  for (unsigned int i = 0; i < ImageType::ImageDimension; i++)
    {
    inputReqRegion.SetSize(i, outputReqRegion.GetSize(i) * m_BinningFactors[i]);
    inputReqRegion.SetIndex(i, inputReqRegion.GetIndex(i) +
                               m_BinningFactors[i] * ( outputReqRegion.GetIndex(i) -
                                                       outputLPRegion.GetIndex(i) ) );
    }
  inputPtr->SetRequestedRegion( inputReqRegion );
}

void BinningImageFilter::GenerateOutputInformation()
{
  const ImageType::SpacingType& inputSpacing    = this->GetInput()->GetSpacing();
  const ImageType::SizeType&    inputSize       = this->GetInput()->GetLargestPossibleRegion().GetSize();
  const ImageType::PointType&   inputOrigin     = this->GetInput()->GetOrigin();

  ImageType::SpacingType  outputSpacing;
  ImageType::SizeType     outputSize;
  ImageType::IndexType    outputStartIndex;
  ImageType::PointType    outputOrigin;

  for (unsigned int i = 0; i < ImageType::ImageDimension; i++)
    {
    outputSpacing[i] = inputSpacing[i] * m_BinningFactors[i];
    outputOrigin[i]  = inputOrigin[i] - 0.5*inputSpacing[i] + 0.5*outputSpacing[i];
    if(inputSize[i]%m_BinningFactors[i] != 0)
      {
      itkExceptionMacro(<< "Binning currently works only for integer divisions")
      }
    else
      outputSize[i] = inputSize[i] / m_BinningFactors[i];

    outputStartIndex[i] = 0;
  }

  this->GetOutput()->SetSpacing( outputSpacing );
  this->GetOutput()->SetOrigin( outputOrigin );

  // Set region
  ImageType::RegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion.SetSize( outputSize );
  outputLargestPossibleRegion.SetIndex( outputStartIndex );
  this->GetOutput()->SetLargestPossibleRegion( outputLargestPossibleRegion );
}

void BinningImageFilter
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType itkNotUsed(threadId) )
{
  const ImageType::PixelType *pIn  = this->GetInput()->GetBufferPointer();
  ImageType::PixelType *pOut = this->GetOutput()->GetBufferPointer();

  // Move pointers to beginning of region
  ImageType::RegionType inputRegion;
  for(unsigned int i=0; i<ImageType::ImageDimension; i++)
    {
    pOut += outputRegionForThread.GetIndex(i) * this->GetOutput()->GetOffsetTable()[i];
    pIn  += outputRegionForThread.GetIndex(i) * this->GetInput() ->GetOffsetTable()[i] * m_BinningFactors[i];

    // Account for difference between buffered and largest possible regions
    pOut -= this->GetOutput()->GetOffsetTable()[i] *
            (this->GetOutput()->GetBufferedRegion().GetIndex()[i] -
             this->GetOutput()->GetLargestPossibleRegion().GetIndex()[i]);
    pIn -= this->GetInput()->GetOffsetTable()[i] *
           (this->GetInput()->GetBufferedRegion().GetIndex()[i] -
            this->GetInput()->GetLargestPossibleRegion().GetIndex()[i]);
    }

  // Binning 2x2
  if(m_BinningFactors[0]==2 && m_BinningFactors[1]==2)
    {
    const size_t buffSize = outputRegionForThread.GetNumberOfPixels()*2;
    int *buffer = new int[buffSize];
    for(unsigned int j=0; j<outputRegionForThread.GetSize(1)*2; j++,
                                                                pIn += this->GetInput()->GetOffsetTable()[1])
      for(unsigned int i=0; i<outputRegionForThread.GetSize(0)*2; i += 2, buffer++)
        *buffer = pIn[i] + pIn[i+1];

    buffer -= buffSize; // Back to original position
    for(unsigned int j=0; j<outputRegionForThread.GetSize(1); j++,
                                                              pOut += this->GetOutput()->GetOffsetTable()[1],
                                                              buffer += 2*outputRegionForThread.GetSize(0))
      for(unsigned int i=0; i<outputRegionForThread.GetSize(0); i++)
        pOut[i] = (buffer[i] + buffer[i+outputRegionForThread.GetSize(0)] ) >> 2;

    buffer -= buffSize; // Back to original position
    delete buffer;
    }

  // Binning 2x1
  else if(m_BinningFactors[0]==2 && m_BinningFactors[1]==1)
    {
    for(unsigned int j=0; j<outputRegionForThread.GetSize(1); j++,
                                                              pIn += this->GetInput()->GetOffsetTable()[1])
      for(unsigned int i=0; i<outputRegionForThread.GetSize(0)*2; i += 2, pOut++)
        *pOut = (pIn[i] + pIn[i+1])>>1;
    }

  // Binning 1x2
  else if(m_BinningFactors[0]==1 && m_BinningFactors[1]==2)
    {
    for(unsigned int j=0; j<outputRegionForThread.GetSize(1); j++,
                                                              pOut += this->GetOutput()->GetOffsetTable()[1],
                                                              pIn += 2*this->GetInput()->GetOffsetTable()[1])
      for(unsigned int i=0; i<outputRegionForThread.GetSize(0); i++)
        pOut[i] = (pIn[i] + pIn[i+this->GetInput()->GetOffsetTable()[1]]) >> 1;
    }
  else
    {
    itkExceptionMacro(<< "Binning factors mismatch! Current factors: "
                      << m_BinningFactors[0] << "x"
                      << m_BinningFactors[1] << " "
                      << "accepted modes [2x2] [2x1] and [1x2]")
    }
}

} // end namespace rtk

#endif // __rtkBinningImageFilter_cxx
