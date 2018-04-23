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

#ifndef rtkBackwardDifferenceDivergenceImageFilter_hxx
#define rtkBackwardDifferenceDivergenceImageFilter_hxx
#include "rtkBackwardDifferenceDivergenceImageFilter.h"

#include <itkConstShapedNeighborhoodIterator.h>
#include <itkNeighborhoodInnerProduct.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkNeighborhoodAlgorithm.h>
#include <itkConstantBoundaryCondition.h>
#include <itkOffset.h>
#include <itkProgressReporter.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
BackwardDifferenceDivergenceImageFilter<TInputImage, TOutputImage>
::BackwardDifferenceDivergenceImageFilter()
{
  m_UseImageSpacing = true;

  // default boundary condition
  m_BoundaryCondition = new itk::ConstantBoundaryCondition<TInputImage>();
  m_IsBoundaryConditionOverriden = false;

  // default behaviour is to process all dimensions
  for (unsigned int dim = 0; dim < TInputImage::ImageDimension; dim++)
    {
    m_DimensionsProcessed[dim] = true;
    }
}

template <class TInputImage, class TOutputImage>
BackwardDifferenceDivergenceImageFilter<TInputImage, TOutputImage>
::~BackwardDifferenceDivergenceImageFilter()
{
  delete m_BoundaryCondition;
}

// This should be handled by an itkMacro, but it doesn't seem to work with pointer types
template <class TInputImage, class TOutputImage>
void
BackwardDifferenceDivergenceImageFilter<TInputImage, TOutputImage>
::SetDimensionsProcessed(bool* DimensionsProcessed)
{
  bool Modified=false;
  for (unsigned int dim=0; dim<TInputImage::ImageDimension; dim++)
    {
    if (m_DimensionsProcessed[dim] != DimensionsProcessed[dim])
      {
      m_DimensionsProcessed[dim] = DimensionsProcessed[dim];
      Modified = true;
      }
    }
  if(Modified) this->Modified();
}

template <class TInputImage, class TOutputImage>
void
BackwardDifferenceDivergenceImageFilter<TInputImage, TOutputImage>
::OverrideBoundaryCondition(itk::ImageBoundaryCondition< TInputImage >* boundaryCondition)
{
  delete m_BoundaryCondition;
  m_BoundaryCondition = boundaryCondition;
  m_IsBoundaryConditionOverriden = true;
}

template <class TInputImage, class TOutputImage>
void
BackwardDifferenceDivergenceImageFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // get pointers to the input and output
  typename TInputImage::Pointer inputPtr = const_cast< TInputImage * >( this->GetInput() );
  typename TOutputImage::Pointer outputPtr = this->GetOutput();

  if ( !inputPtr || !outputPtr )
    {
    return;
    }

  // get a copy of the input requested region (should equal the output
  // requested region)
  typename TInputImage::RegionType inputRequestedRegion;
  inputRequestedRegion = inputPtr->GetRequestedRegion();

  // pad the input requested region by the operator radius
  inputRequestedRegion.PadByRadius(1);

  // crop the input requested region at the input's largest possible region
  if ( inputRequestedRegion.Crop( inputPtr->GetLargestPossibleRegion() ) )
    {
    inputPtr->SetRequestedRegion(inputRequestedRegion);
    return;
    }
  else
    {
    // Couldn't crop the region (requested region is outside the largest
    // possible region).  Throw an exception.

    // store what we tried to request (prior to trying to crop)
    inputPtr->SetRequestedRegion(inputRequestedRegion);

    // build an exception
    itk::InvalidRequestedRegionError e(__FILE__, __LINE__);
    e.SetLocation(ITK_LOCATION);
    e.SetDescription("Requested region is (at least partially) outside the largest possible region.");
    e.SetDataObject(inputPtr);
    throw e;
    }
}

template <class TInputImage, class TOutputImage>
void
BackwardDifferenceDivergenceImageFilter< TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  if (m_UseImageSpacing == false)
    {
    m_InvSpacingCoeffs.Fill(1.0);
    }
  else
    {
    m_InvSpacingCoeffs= this->GetInput()->GetSpacing();
    for (unsigned int dim = 0; dim < TInputImage::ImageDimension; dim++)
      {
      m_InvSpacingCoeffs[dim] = 1.0/m_InvSpacingCoeffs[dim];
      }
    }
}

template <class TInputImage, class TOutputImage>
void
BackwardDifferenceDivergenceImageFilter< TInputImage, TOutputImage>
::ThreadedGenerateData(const typename TInputImage::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  // Generate a list of indices of the dimensions to process
  std::vector<int> dimsToProcess;
  dimsToProcess.reserve(TInputImage::ImageDimension);
  for (unsigned int dim = 0; dim < TInputImage::ImageDimension; dim++)
    {
    if(m_DimensionsProcessed[dim]) dimsToProcess.push_back(dim);
    }

  typename TOutputImage::Pointer output = this->GetOutput();
  typename TInputImage::ConstPointer input = this->GetInput();

  itk::ImageRegionIterator<TOutputImage> oit(output, outputRegionForThread);
  oit.GoToBegin();

  itk::Size<InputImageDimension> radius;
  radius.Fill(1);

  itk::ConstNeighborhoodIterator<TInputImage> iit(radius, input, outputRegionForThread);
  iit.GoToBegin();
  iit.OverrideBoundaryCondition(m_BoundaryCondition);

  const itk::SizeValueType c = (itk::SizeValueType) (iit.Size() / 2); // get offset of center pixel
  itk::SizeValueType strides[TOutputImage::ImageDimension]; // get offsets to access neighboring pixels
  for (unsigned int dim=0; dim<TOutputImage::ImageDimension; dim++)
    {
    strides[dim] = iit.GetStride(dim);
    }

  while(!oit.IsAtEnd())
    {
    typename TOutputImage::PixelType div = 0.0F;
    // Compute the local differences around the central pixel
    for (unsigned int k = 0; k < dimsToProcess.size(); k++)
      {
      div += (iit.GetPixel(c)[k] - iit.GetPixel(c - strides[dimsToProcess[k]])[k]) * m_InvSpacingCoeffs[dimsToProcess[k]];
      }
    oit.Set(div);
    ++oit;
    ++iit;
    }
}

template <class TInputImage, class TOutputImage>
void
BackwardDifferenceDivergenceImageFilter< TInputImage, TOutputImage>
::AfterThreadedGenerateData()
{
  if (m_IsBoundaryConditionOverriden) return;

  std::vector<int> dimsToProcess;
  for (unsigned int dim = 0; dim < TInputImage::ImageDimension; dim++)
    {
    if(m_DimensionsProcessed[dim]) dimsToProcess.push_back(dim);
    }

  // The conditions on the borders this filter requires are very specific
  // Some must be enforced explicitely as they do not correspond to any of the
  // padding styles available in ITK.
  // This needs to be performed only if the output requested region contains
  // the borders of the image.
  // It is ignored if the boundary condition has been overriden (this function returns before)

  typename TOutputImage::RegionType largest = this->GetOutput()->GetLargestPossibleRegion();

  for (unsigned int k=0; k<dimsToProcess.size(); k++)
    {
    // Create a slice region at the border of the largest possible region
    typename TOutputImage::RegionType slice = largest;
    slice.SetSize(dimsToProcess[k], 1);
    slice.SetIndex(dimsToProcess[k], largest.GetSize()[dimsToProcess[k]] - 1);

    // If it overlaps the output buffered region, enforce boundary condition
    // on the overlap
    if ( slice.Crop( this->GetOutput()->GetBufferedRegion() ) )
      {
      itk::ImageRegionIterator<TOutputImage> oit(this->GetOutput(), slice);
      itk::ImageRegionConstIterator<TInputImage> iit(this->GetInput(), slice);

      oit.Set(oit.Get() - iit.Get()[k] * m_InvSpacingCoeffs[dimsToProcess[k]]);
      ++oit;
      ++iit;
      }
    }
}


} // end namespace rtk

#endif
