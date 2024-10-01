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

#ifndef rtkKatsevichDerivativeImageFilter_hxx
#define rtkKatsevichDerivativeImageFilter_hxx

#include "rtkKatsevichDerivativeImageFilter.h"

#include "itkNeighborhoodOperatorImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkProgressAccumulator.h"
#include "itkObjectFactory.h"


namespace rtk
{

/**
 *   Constructor
 */
template <typename TInputImage, typename TOutputImage>
KatsevichDerivativeImageFilter<TInputImage, TOutputImage>::KatsevichDerivativeImageFilter()
{
  this->SetNumberOfRequiredInputs(1);
}

template <class TInputImage, class TOutputImage>
void
KatsevichDerivativeImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() ITKv5_CONST
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull() || !this->m_Geometry->GetTheGeometryIsVerified())
    itkExceptionMacro(<< "Geometry has not been set or not been checked");
}

template <typename TInputImage, typename TOutputImage>
void
KatsevichDerivativeImageFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
{
  // Update output size and origin
  // Call the superclass' implementation of this method
  Superclass::GenerateOutputInformation();

  // Get pointers to the input and output
  const InputImageType * inputPtr = this->GetInput();
  OutputImageType *      outputPtr = this->GetOutput();

  itkAssertInDebugAndIgnoreInReleaseMacro(inputPtr);
  itkAssertInDebugAndIgnoreInReleaseMacro(outputPtr != nullptr);

  const typename TInputImage::SpacingType & inputSpacing = inputPtr->GetSpacing();
  const typename TInputImage::PointType &   inputOrigin = inputPtr->GetOrigin();
  const typename TInputImage::SizeType &    inputSize = inputPtr->GetLargestPossibleRegion().GetSize();
  const typename TInputImage::IndexType &   inputStartIndex = inputPtr->GetLargestPossibleRegion().GetIndex();
  typename TOutputImage::IndexType          outputStartIndex;

  typename TOutputImage::SpacingType outputSpacing(inputSpacing);
  typename TOutputImage::SizeType    outputSize(inputSize);
  typename TOutputImage::PointType   outputOrigin(inputOrigin);

  for (int i = 0; i < 3; i++)
  {
    outputOrigin[i] += 0.5 * inputSpacing[i];
    // If cylindrical detector, sampling in v is unchanged. In case of a flat panel
    // derivative is also computed at interlaced positions in v.
    if (!(this->m_Geometry->GetRadiusCylindricalDetector() != 0. && i == 1))
    {
      outputSize[i] -= 1;
    }
    if (outputSize[i] < 1)
    {
      itkExceptionMacro(<< "Output size is 0 on dimension " << i);
    }
  }

  typename TOutputImage::RegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion.SetIndex(inputStartIndex);
  outputLargestPossibleRegion.SetSize(outputSize);
  outputPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);
  outputPtr->SetOrigin(outputOrigin);
  outputPtr->SetSpacing(outputSpacing);
}

template <typename TInputImage, typename TOutputImage>
void
KatsevichDerivativeImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();
}

template <typename TInputImage, typename TOutputImage>
void
KatsevichDerivativeImageFilter<TInputImage, TOutputImage>::GenerateData()
{

  // Get pointers to the input and output
  const InputImageType * inputPtr = this->GetInput();
  OutputImageType *      outputPtr = this->GetOutput();

  if (this->m_Geometry->GetRadiusCylindricalDetector() != 0) // Curved detector
  {
    // ShapedIterator on InputImage
    typename ConstShapedNeighborhoodIteratorType::OffsetType offset1 = { { 1, 0, 1 } };
    typename ConstShapedNeighborhoodIteratorType::OffsetType offset2 = { { 1, 0, 0 } };
    typename ConstShapedNeighborhoodIteratorType::OffsetType offset3 = { { 0, 0, 1 } };
    typename ConstShapedNeighborhoodIteratorType::OffsetType offset4 = { { 0, 0, 0 } };

    typename ConstShapedNeighborhoodIteratorType::RadiusType radius;
    radius.Fill(1);

    ConstShapedNeighborhoodIteratorType itIn(radius, inputPtr, inputPtr->GetRequestedRegion());
    itIn.ActivateOffset(offset1);
    itIn.ActivateOffset(offset2);
    itIn.ActivateOffset(offset3);
    itIn.ActivateOffset(offset4);

    outputPtr->SetRegions(outputPtr->GetLargestPossibleRegion());
    outputPtr->Allocate();

    IteratorType itOut(outputPtr, outputPtr->GetRequestedRegion());

    typename InputImageType::SpacingType spacing = inputPtr->GetSpacing();
    typename InputImageType::PointType   origin = inputPtr->GetOrigin();
    typename InputImageType::PixelType   angular_gap = this->m_Geometry->GetHelixAngularGap();
    float                                D = this->m_Geometry->GetHelixSourceToDetectorDist();

    for (itIn.GoToBegin(), itOut.GoToBegin(); !itIn.IsAtEnd(); ++itIn, ++itOut)
    {
      OutputPixelType t1 = 0.5 * (1 / spacing[0] + 1 / angular_gap) * itIn.GetPixel(offset1);
      OutputPixelType t2 = 0.5 * (1 / spacing[0] - 1 / angular_gap) * itIn.GetPixel(offset2);
      OutputPixelType t3 = 0.5 * (-1 / spacing[0] + 1 / angular_gap) * itIn.GetPixel(offset3);
      OutputPixelType t4 = 0.5 * (-1 / spacing[0] - 1 / angular_gap) * itIn.GetPixel(offset4);

      // Length-correction weighting
      typename InputImageType::IndexType index = itIn.GetIndex();
      OutputPixelType                    v = origin[1] + spacing[1] * index[1];

      OutputPixelType length_correction = D / (sqrt(pow(D, 2) + pow(v, 2)));
      itOut.Set(length_correction * (t1 + t2 + t3 + t4));
    }
  }
  else // Flat panel detector
  {

    // ShapedIterator on InputImage
    typename ConstShapedNeighborhoodIteratorType::OffsetType offset000 = { { 0, 0, 0 } };
    typename ConstShapedNeighborhoodIteratorType::OffsetType offset100 = { { 1, 0, 0 } };
    typename ConstShapedNeighborhoodIteratorType::OffsetType offset010 = { { 0, 1, 0 } };
    typename ConstShapedNeighborhoodIteratorType::OffsetType offset001 = { { 0, 0, 1 } };
    typename ConstShapedNeighborhoodIteratorType::OffsetType offset110 = { { 1, 1, 0 } };
    typename ConstShapedNeighborhoodIteratorType::OffsetType offset101 = { { 1, 0, 1 } };
    typename ConstShapedNeighborhoodIteratorType::OffsetType offset011 = { { 0, 1, 1 } };
    typename ConstShapedNeighborhoodIteratorType::OffsetType offset111 = { { 1, 1, 1 } };

    typename ConstShapedNeighborhoodIteratorType::RadiusType radius;
    radius.Fill(1);

    ConstShapedNeighborhoodIteratorType itIn(radius, inputPtr, inputPtr->GetRequestedRegion());
    itIn.ActivateOffset(offset000);
    itIn.ActivateOffset(offset100);
    itIn.ActivateOffset(offset010);
    itIn.ActivateOffset(offset001);
    itIn.ActivateOffset(offset110);
    itIn.ActivateOffset(offset101);
    itIn.ActivateOffset(offset011);
    itIn.ActivateOffset(offset111);

    outputPtr->SetRegions(outputPtr->GetLargestPossibleRegion());
    outputPtr->Allocate();

    IteratorType itOut(outputPtr, outputPtr->GetRequestedRegion());

    typename InputImageType::SpacingType spacing = inputPtr->GetSpacing();
    typename InputImageType::PointType   origin = inputPtr->GetOrigin();
    typename InputImageType::PixelType   angular_gap = this->m_Geometry->GetHelixAngularGap();
    float                                D = this->m_Geometry->GetHelixSourceToDetectorDist();

    for (itIn.GoToBegin(), itOut.GoToBegin(); !itIn.IsAtEnd(); ++itIn, ++itOut)
    {
      OutputPixelType p000 = itIn.GetPixel(offset000);
      OutputPixelType p100 = itIn.GetPixel(offset100);
      OutputPixelType p010 = itIn.GetPixel(offset010);
      OutputPixelType p001 = itIn.GetPixel(offset001);
      OutputPixelType p110 = itIn.GetPixel(offset110);
      OutputPixelType p101 = itIn.GetPixel(offset101);
      OutputPixelType p011 = itIn.GetPixel(offset011);
      OutputPixelType p111 = itIn.GetPixel(offset111);

      typename InputImageType::IndexType index = itIn.GetIndex();
      OutputPixelType u = origin[0] + spacing[0] * (index[0] + 0.5); // Add half-pixel for interlaced positions.
      OutputPixelType v = origin[1] + spacing[1] * (index[1] + 0.5); //


      OutputPixelType dl = 0.25 * (1 / angular_gap) * (p001 - p000 + p011 - p010 + p101 - p100 + p111 - p110);
      OutputPixelType dutmp = 0.25 * (1 / spacing[0]) * (p100 - p000 + p110 - p010 + p101 - p001 + p111 - p011);
      OutputPixelType dvtmp = 0.25 * (1 / spacing[1]) * (p010 - p000 + p110 - p100 + p011 - p001 + p111 - p101);

      OutputPixelType length_correction = D / (sqrt(pow(D, 2) + pow(u, 2) + pow(v, 2)));
      itOut.Set(length_correction * (dl + (pow(u, 2) + pow(D, 2)) / D * dutmp + u * v / D * dvtmp));
    }
  }

  this->GraftOutput(outputPtr);

  // Update geometry to shift gantry angle by half angular step
  rtk::ThreeDHelicalProjectionGeometry::Pointer new_geometry = rtk::ThreeDHelicalProjectionGeometry::New();
  const std::vector<double>                     angles = this->GetGeometry()->GetHelicalAngles();
  const double                                  angular_gap = this->GetGeometry()->GetHelixAngularGap();
  const double                                  sid = this->GetGeometry()->GetHelixRadius();
  const double                                  sdd = this->GetGeometry()->GetHelixSourceToDetectorDist();
  const std::vector<double>                     dy = this->GetGeometry()->GetSourceOffsetsY();
  const double                                  vertical_gap = this->GetGeometry()->GetHelixVerticalGap();

  double new_angle(0);
  double new_dy(0);

  for (int i = 0; i < angles.size(); i++)
  {
    new_angle = angles[i] + angular_gap * 0.5;
    new_dy = dy[i] + vertical_gap * 0.5;
    new_geometry->AddProjectionInRadians(sid, sdd, new_angle, 0, new_dy, 0, 0, 0, new_dy);
  }
  new_geometry->VerifyHelixParameters();
  this->SetGeometry(new_geometry);
}


template <typename TInputImage, typename TOutputImage>
void
KatsevichDerivativeImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

} // end namespace rtk

#endif
