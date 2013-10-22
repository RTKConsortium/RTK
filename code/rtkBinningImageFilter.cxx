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
  const ImageType::IndexType&   inputStartIndex = this->GetInput()->GetLargestPossibleRegion().GetIndex();
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
::ThreadedGenerateData(const OutputImageRegionType& itkNotUsed(outputRegionForThread), ThreadIdType threadId )
{

  int inputSize[2], binSize[2];
  inputSize[0] = this->GetInput()->GetLargestPossibleRegion().GetSize()[0];
  inputSize[1] = this->GetInput()->GetLargestPossibleRegion().GetSize()[1];
  binSize[0] = (inputSize[0]>>1);
  binSize[1] = (inputSize[1]>>1);

  typedef   ImageType::PixelType inputPixel;
  typedef   ImageType::PixelType outputPixel;

  //this->GenerateOutputInformation();
  const inputPixel  *bufferIn  = this->GetInput()->GetBufferPointer();
  outputPixel *bufferOut = this->GetOutput()->GetBufferPointer();

  int initX = threadId*inputSize[0]*inputSize[1]/(this->GetNumberOfThreads()*m_BinningFactors[0]);
  int endX = initX + binSize[0]*inputSize[1]/this->GetNumberOfThreads();
  int initY = threadId*binSize[1]/this->GetNumberOfThreads();
  int endY = initY + binSize[1]/this->GetNumberOfThreads();

  // Binning 2x2
  if(m_BinningFactors[0]==2 && m_BinningFactors[1]==2)
  {
    const int Vlength = endX-initX;
    std::vector<int> V(Vlength,0);

    int i, j, idx, pos, vj;
    for (i=0, idx=initX*m_BinningFactors[0]; i < Vlength; i++, idx+=2) // Only works if number of pixels per line multiple of 2
    {
        V[i] = ( (bufferIn[idx])+(bufferIn[idx+1]) );
    }
    for (j=initY, vj=0; j<endY; j++, vj++)
    {
      pos = j*binSize[0];
      for (i=0; i < binSize[0]; i++, pos++)
        {
            bufferOut[pos] = ( (V[2*vj*binSize[0]+i])+(V[(2*vj+1)*binSize[0]+i]) )>>2;
        }
    }
  }

  // Binning 2x1
  else if(m_BinningFactors[0]==2 && m_BinningFactors[1]==1)
  {
    int i, idx;
    for (i=initX, idx=initX*m_BinningFactors[0]; i < endX; i++, idx+=2) // Only works if number of pixels per line multiple of 2
        bufferOut[i] = ( (bufferIn[idx])+(bufferIn[idx+1]) )>>1;
  }

  // Binning 1x2
  else if(m_BinningFactors[0]==1 && m_BinningFactors[1]==2)
  {
    int i, j, pos;
    for (j=initY; j<endY;j++)
    {
      pos = j*inputSize[0];
      for (i=0; i<inputSize[0];i++, pos++)
        bufferOut[pos] = ( (bufferIn[2*j*inputSize[0]+i])+(bufferIn[(2*j+1)*inputSize[0]+i]) )>>1;
    }
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
