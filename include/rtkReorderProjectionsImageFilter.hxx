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

#ifndef rtkReorderProjectionsImageFilter_hxx
#define rtkReorderProjectionsImageFilter_hxx

#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
ReorderProjectionsImageFilter<TInputImage, TOutputImage>
::ReorderProjectionsImageFilter()
{
  m_OutputGeometry = GeometryType::New();
}

template <class TInputImage, class TOutputImage>
void
ReorderProjectionsImageFilter<TInputImage, TOutputImage>
::SetInputSignal(const std::vector<double> signal)
{
  m_InputSignal = signal;
}

template <class TInputImage, class TOutputImage>
std::vector<double>
ReorderProjectionsImageFilter<TInputImage, TOutputImage>
::GetOutputSignal()
{
  return m_OutputSignal;
}

template <class TInputImage, class TOutputImage>
void
ReorderProjectionsImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  std::vector<unsigned int> permutation = rtk::GetSortingPermutation(m_InputSignal);

  // Allocate the pixels of the output, and at first fill them with zeros
  this->GetOutput()->SetBufferedRegion(this->GetOutput()->GetRequestedRegion());
  this->GetOutput()->Allocate();
  this->GetOutput()->FillBuffer(0);

  // Declare regions used in the loop
  typename TInputImage::RegionType inputRegion = this->GetOutput()->GetRequestedRegion();
  typename TInputImage::RegionType outputRegion = this->GetOutput()->GetRequestedRegion();
  inputRegion.SetSize(2, 1);
  outputRegion.SetSize(2, 1);

  // Initialize objects (otherwise, if the filter runs several times,
  // the outputs become incorrect)
  m_OutputGeometry->Clear();
  m_OutputSignal.clear();

  // Perform the copies
  for (unsigned int proj=0; proj<this->GetOutput()->GetRequestedRegion().GetSize()[2]; proj++)
    {
    // Copy the projection data

    // Regions
    inputRegion.SetIndex(2, permutation[proj]);
    outputRegion.SetIndex(2, proj);

    itk::ImageRegionConstIterator<TInputImage> inputProjsIt(this->GetInput(), inputRegion);
    itk::ImageRegionIterator<TOutputImage> outputProjsIt(this->GetOutput(), outputRegion);

    // Actual copy
    while(!outputProjsIt.IsAtEnd())
      {
      outputProjsIt.Set(inputProjsIt.Get());
      ++outputProjsIt;
      ++inputProjsIt;
      }

    // Copy the geometry
    m_OutputGeometry->SetRadiusCylindricalDetector(m_InputGeometry->GetRadiusCylindricalDetector());
    m_OutputGeometry->AddProjectionInRadians(m_InputGeometry->GetSourceToIsocenterDistances()[permutation[proj]],
                                             m_InputGeometry->GetSourceToDetectorDistances()[permutation[proj]],
                                             m_InputGeometry->GetGantryAngles()[permutation[proj]],
                                             m_InputGeometry->GetProjectionOffsetsX()[permutation[proj]],
                                             m_InputGeometry->GetProjectionOffsetsY()[permutation[proj]],
                                             m_InputGeometry->GetOutOfPlaneAngles()[permutation[proj]],
                                             m_InputGeometry->GetInPlaneAngles()[permutation[proj]],
                                             m_InputGeometry->GetSourceOffsetsX()[permutation[proj]],
                                             m_InputGeometry->GetSourceOffsetsY()[permutation[proj]]);
    m_OutputGeometry->SetCollimationOfLastProjection(m_InputGeometry->GetCollimationUInf()[permutation[proj]],
                                                     m_InputGeometry->GetCollimationUSup()[permutation[proj]],
                                                     m_InputGeometry->GetCollimationVInf()[permutation[proj]],
                                                     m_InputGeometry->GetCollimationVSup()[permutation[proj]]);

    // Copy the signal
    m_OutputSignal.push_back(m_InputSignal[permutation[proj]]);
    }

}

} // end namespace rtk
#endif
