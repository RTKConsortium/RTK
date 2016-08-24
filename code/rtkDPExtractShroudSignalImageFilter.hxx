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

#ifndef rtkDPExtractShroudSignalImageFilter_hxx
#define rtkDPExtractShroudSignalImageFilter_hxx

#include <itkArray.h>

namespace rtk
{

template<class TInputPixel, class TOutputPixel>
DPExtractShroudSignalImageFilter<TInputPixel, TOutputPixel>
::DPExtractShroudSignalImageFilter() :
  m_Amplitude(0.)
{
}

template<class TInputPixel, class TOutputPixel>
void
DPExtractShroudSignalImageFilter<TInputPixel, TOutputPixel>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer  inputPtr =
    const_cast< TInputImage * >( this->GetInput() );
  if ( !inputPtr )
    {
    return;
    }
  inputPtr->SetRequestedRegion(inputPtr->GetLargestPossibleRegion());
}

template<class TInputPixel, class TOutputPixel>
void
DPExtractShroudSignalImageFilter<TInputPixel, TOutputPixel>
::GenerateOutputInformation()
{
  // get pointers to the input and output
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput();

  if ( !outputPtr || !inputPtr)
  {
    return;
  }
  typename TOutputImage::RegionType outRegion;
  typename TOutputImage::RegionType::SizeType outSize;
  typename TOutputImage::RegionType::IndexType outIdx;
  outSize[0] = this->GetInput()->GetLargestPossibleRegion().GetSize()[1];
  outIdx[0] = this->GetInput()->GetLargestPossibleRegion().GetIndex()[1];
  outRegion.SetSize(outSize);
  outRegion.SetIndex(outIdx);

  const typename TInputImage::SpacingType &
    inputSpacing = inputPtr->GetSpacing();
  typename TOutputImage::SpacingType outputSpacing;
  outputSpacing[0] = inputSpacing[1];
  outputPtr->SetSpacing(outputSpacing);

  typename TOutputImage::DirectionType outputDirection;
  outputDirection[0][0] = 1;
  outputPtr->SetDirection(outputDirection);

  const typename TInputImage::PointType &
    inputOrigin = inputPtr->GetOrigin();
  typename TOutputImage::PointType outputOrigin;
  outputOrigin[0] = inputOrigin[1];
  outputPtr->SetOrigin(outputOrigin);

  outputPtr->SetLargestPossibleRegion(outRegion);
}

template<class TInputPixel, class TOutputPixel>
void
DPExtractShroudSignalImageFilter<TInputPixel, TOutputPixel>
::GenerateData()
{
  this->AllocateOutputs();

  typename TInputImage::ConstPointer input = this->GetInput();
  typename TInputImage::RegionType::IndexType inputIdx = input->GetLargestPossibleRegion().GetIndex();
  typename TInputImage::RegionType::SizeType  inputSize = input->GetLargestPossibleRegion().GetSize();

  typename itk::Image<int, InputImageDimension>::Pointer from = itk::Image<int, InputImageDimension>::New();
  from->CopyInformation(input);
  from->SetRegions(input->GetLargestPossibleRegion());
  from->Allocate();
  int amplitudeInVoxel = m_Amplitude / input->GetSpacing()[0];

  itk::Array<double>* prev = new itk::Array<double>(inputSize[0]);
  itk::Array<double>* curr = new itk::Array<double>(inputSize[0]);

  typename TInputImage::RegionType::IndexType idx = inputIdx;
  for (unsigned i = 0; i < inputSize[0]; ++i)
  {
    (*prev)[i] = (*input)[idx];
    idx[0]++;
  }

  for (unsigned i = 1; i < inputSize[1]; ++i)
  {
    (*curr).Fill(0.0);
    idx[0] = inputIdx[0];
    for (unsigned j = 0; j < inputSize[0]; ++j)
    {
      for (int r = -amplitudeInVoxel; r <= amplitudeInVoxel; ++r)
      {
        typename TInputImage::RegionType::IndexType idx_to = idx;
        idx_to[0] += r;
        idx_to[1]++;
        if (input->GetLargestPossibleRegion().IsInside(idx_to) && (*input)[idx_to] + (*prev)[j] > (*curr)[j + r])
        {
          (*curr)[j + r] = (*input)[idx_to] + (*prev)[j];
          (*from)[idx_to] = -r;
        }
      }
      idx[0]++;
    }
    idx[1]++;
    std::swap(prev, curr);
  }

  unsigned max = 0;
  for (unsigned j = 1; j < inputSize[0]; ++j)
    if ((*prev)[j] > (*prev)[max])
      max = j;

  idx[0] = inputIdx[0] + max;

  typename Superclass::OutputImagePointer output = this->GetOutput();
  output->Allocate();
  typename TOutputImage::RegionType::IndexType outputIdx;
  TOutputPixel value = 0;
  outputIdx[0] = idx[1];
  (*output)[outputIdx] = value;
  while (idx[1] != inputIdx[1])
  {
    outputIdx[0]--;
    value -= (*from)[idx] * input->GetSpacing()[0];
    (*output)[outputIdx] = value;
    idx[0] += (*from)[idx];
    idx[1]--;
  }
  delete prev;
  delete curr;
}

} // end of namespace rtk
#endif
