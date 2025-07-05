/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkDisplacedDetectorImageFilter_hxx
#define rtkDisplacedDetectorImageFilter_hxx


#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
void
DisplacedDetectorImageFilter<TInputImage, TOutputImage>::SetOffsets(double minOffset, double maxOffset)
{
  m_OffsetsSet = true;
  itkDebugMacro("setting MinimumOffset to " << minOffset);
  if (this->m_MinimumOffset != minOffset)
  {
    this->m_MinimumOffset = minOffset;
    this->Modified();
  }
  itkDebugMacro("setting MaximumOffset to " << maxOffset);
  if (this->m_MaximumOffset != maxOffset)
  {
    this->m_MaximumOffset = maxOffset;
    this->Modified();
  }
}

template <class TInputImage, class TOutputImage>
void
DisplacedDetectorImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() const
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull())
    itkExceptionMacro(<< "Geometry has not been set.");
}

/**
 * Account for the padding computed in GenerateOutputInformation to propagate the
 * requested region.
 */
template <class TInputImage, class TOutputImage>
void
DisplacedDetectorImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer  inputPtr = const_cast<TInputImage *>(this->GetInput());
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();

  if (!inputPtr || !outputPtr)
    return;

  typename TInputImage::RegionType inputRequestedRegion = outputPtr->GetRequestedRegion();
  inputRequestedRegion.Crop(inputPtr->GetLargestPossibleRegion());

  inputPtr->SetRequestedRegion(inputRequestedRegion);
}

/**
 * When the detector is displaced, one needs to zero pad the input data on the
 * nearest side to the center.
 */
template <class TInputImage, class TOutputImage>
void
DisplacedDetectorImageFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
{
  // get pointers to the input and output
  typename Superclass::InputImagePointer  inputPtr = const_cast<TInputImage *>(this->GetInput());
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();

  if (!outputPtr || !inputPtr)
  {
    return;
  }

  // Copy the meta data for this data type
  outputPtr->SetSpacing(inputPtr->GetSpacing());
  outputPtr->SetOrigin(inputPtr->GetOrigin());
  outputPtr->SetDirection(inputPtr->GetDirection());
  outputPtr->SetNumberOfComponentsPerPixel(inputPtr->GetNumberOfComponentsPerPixel());

  typename TOutputImage::RegionType outputLargestPossibleRegion = inputPtr->GetLargestPossibleRegion();

  if (m_Disable)
  {
    this->SetInPlace(true);
    outputPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);
    return;
  }
  else if (this->GetGeometry()->GetRadiusCylindricalDetector() != 0)
  {
    itkGenericExceptionMacro(<< "Displaced detector cannot handle cylindrical detector. "
                             << "Consider disabling it by setting m_Disable=true "
                             << "or using the nodisplaced flag of the application you are running");
  }

  // Compute the X coordinates of the corners of the image
  typename Superclass::InputImageType::PointType corner;
  inputPtr->TransformIndexToPhysicalPoint(inputPtr->GetLargestPossibleRegion().GetIndex(), corner);
  m_InferiorCorner = corner[0];
  m_SuperiorCorner = m_InferiorCorner;
  if (inputPtr->GetSpacing()[0] < 0.)
    m_InferiorCorner += inputPtr->GetSpacing()[0] * (outputLargestPossibleRegion.GetSize(0) - 1);
  else
    m_SuperiorCorner += inputPtr->GetSpacing()[0] * (outputLargestPossibleRegion.GetSize(0) - 1);

  if (!m_OffsetsSet)
  {
    // Account for projections offsets
    double minSupUntiltCorner = itk::NumericTraits<double>::max();
    double maxInfUntiltCorner = itk::NumericTraits<double>::NonpositiveMin();
    for (unsigned int i = 0; i < m_Geometry->GetProjectionOffsetsX().size(); i++)
    {
      maxInfUntiltCorner =
        std::max(maxInfUntiltCorner, m_Geometry->ToUntiltedCoordinateAtIsocenter(i, m_InferiorCorner));
      minSupUntiltCorner =
        std::min(minSupUntiltCorner, m_Geometry->ToUntiltedCoordinateAtIsocenter(i, m_SuperiorCorner));
    }
    m_InferiorCorner = maxInfUntiltCorner;
    m_SuperiorCorner = minSupUntiltCorner;
  }
  else
  {
    m_InferiorCorner += m_MaximumOffset;
    m_SuperiorCorner += m_MinimumOffset;
  }

  // 4 cases depending on the position of the two corners
  // Case 1: Impossible to account for too large displacements
  if (m_InferiorCorner > 0. || m_SuperiorCorner < 0.)
  {
    itkGenericExceptionMacro(<< "Cannot account for detector displacement larger than 50% of panel size."
                             << " Corner inf=" << m_InferiorCorner << " and corner sup=" << m_SuperiorCorner);
  }
  // Case 2: Not displaced, or explicit request not to pad: default outputLargestPossibleRegion is fine
  else if ((itk::Math::abs(m_InferiorCorner + m_SuperiorCorner) <
            0.1 * itk::Math::abs(m_SuperiorCorner - m_InferiorCorner)) ||
           !m_PadOnTruncatedSide)
  {
    this->SetInPlace(true);
  }
  else if (m_SuperiorCorner + m_InferiorCorner > 0.)
  {
    this->SetInPlace(false);
    typename itk::Index<NDimension>::IndexValueType index =
      outputLargestPossibleRegion.GetIndex()[0] - outputLargestPossibleRegion.GetSize()[0];
    outputLargestPossibleRegion.SetIndex(0, index);
    outputLargestPossibleRegion.SetSize(0, outputLargestPossibleRegion.GetSize()[0] * 2);
  }
  else
  {
    this->SetInPlace(false);
    outputLargestPossibleRegion.SetSize(0, outputLargestPossibleRegion.GetSize()[0] * 2);
  }

  outputPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);
}

template <class TInputImage, class TOutputImage>
void
DisplacedDetectorImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread)
{
  // Declare the pixel and initialize it to the right length (for vector images)
  typename TOutputImage::PixelType pix;
  itk::NumericTraits<typename TOutputImage::PixelType>::SetLength(pix,
                                                                  this->GetInput()->GetNumberOfComponentsPerPixel());

  // Compute overlap between input and output
  itk::ImageRegionIterator<OutputImageType> itOut(this->GetOutput(), outputRegionForThread);
  OutputImageRegionType                     overlapRegion = outputRegionForThread;
  if (!overlapRegion.Crop(this->GetInput()->GetLargestPossibleRegion()))
  {
    // No overlap, set output region to 0
    while (!itOut.IsAtEnd())
    {
      itOut.Set({});
      ++itOut;
    }
    return;
  }
  itk::ImageRegionConstIterator<InputImageType> itIn(this->GetInput(), overlapRegion);

  // Not displaced, nothing to do
  if ((itk::Math::abs(m_InferiorCorner + m_SuperiorCorner) <
       0.1 * itk::Math::abs(m_SuperiorCorner - m_InferiorCorner)) ||
      m_Disable)
  {
    // If not in place, copy is required
    if (this->GetInput() != this->GetOutput())
    {
      while (!itIn.IsAtEnd())
      {
        itOut.Set(itIn.Get());
        ++itIn;
        ++itOut;
      }
    }
    return;
  }

  // Weight image parameters
  typename WeightImageType::RegionType  region;
  typename WeightImageType::SpacingType spacing;
  typename WeightImageType::PointType   origin;
  region.SetSize(0, overlapRegion.GetSize(0));
  region.SetIndex(0, overlapRegion.GetIndex(0));
  spacing[0] = this->GetInput()->GetSpacing()[0];
  origin[0] = this->GetInput()->GetOrigin()[0];

  // Create one line of weights
  auto weights = WeightImageType::New();
  weights->SetSpacing(spacing);
  weights->SetOrigin(origin);
  weights->SetRegions(region);
  weights->Allocate();
  typename itk::ImageRegionIteratorWithIndex<WeightImageType> itWeights(weights, weights->GetLargestPossibleRegion());

  double       theta = std::min(-1 * m_InferiorCorner, m_SuperiorCorner);
  unsigned int nLinesPerProjection = overlapRegion.GetNumberOfPixels();
  nLinesPerProjection /= overlapRegion.GetSize(0);
  nLinesPerProjection /= overlapRegion.GetSize(NDimension - 1);
  for (unsigned int k = 0; k < overlapRegion.GetSize(NDimension - 1); k++)
  {
    // Prepare weights for current slice (depends on ProjectionOffsetsX)
    const double sx = m_Geometry->GetSourceOffsetsX()[itIn.GetIndex()[NDimension - 1]];
    double       sid = m_Geometry->GetSourceToIsocenterDistances()[itIn.GetIndex()[NDimension - 1]];
    sid = sqrt(sid * sid + sx * sx); // To untilted situation
    double invsid = 0.;
    double invden = 0.;
    if (sid != 0.)
    {
      invsid = 1. / sid;
      invden = 1. / (2. * std::atan(theta * invsid));
    }
    typename WeightImageType::PointType point;
    weights->TransformIndexToPhysicalPoint(itWeights.GetIndex(), point);

    if (m_SuperiorCorner + m_InferiorCorner > 0.)
    {
      itWeights.GoToBegin();
      while (!itWeights.IsAtEnd())
      {
        const double l = m_Geometry->ToUntiltedCoordinateAtIsocenter(itIn.GetIndex()[NDimension - 1], point[0]);
        if (l <= -1 * theta)
          itWeights.Set(0.0);
        else if (l >= theta)
          itWeights.Set(2.0);
        else
          itWeights.Set(sin(itk::Math::pi * atan(l * invsid) * invden) + 1);
        ++itWeights;
        point[0] += spacing[0];
      }
    }
    else
    {
      while (!itWeights.IsAtEnd())
      {
        const double l = m_Geometry->ToUntiltedCoordinateAtIsocenter(itIn.GetIndex()[NDimension - 1], point[0]);
        if (l <= -1 * theta)
          itWeights.Set(2.0);
        else if (l >= theta)
          itWeights.Set(0.0);
        else
          itWeights.Set(1 - sin(itk::Math::pi * atan(l * invsid) * invden));
        ++itWeights;
        point[0] += spacing[0];
      }
    }

    // Multiply each line of the current 2D or 3D slice
    for (unsigned int j = 0; j < nLinesPerProjection; j++)
    {
      // Set outside of overlap to 0 values
      while (itOut.GetIndex()[0] != itIn.GetIndex()[0])
      {
        pix = itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue(pix);
        itOut.Set(pix);
        ++itOut;
      }

      itWeights.GoToBegin();
      while (!itWeights.IsAtEnd())
      {
        itOut.Set(itIn.Get() * itWeights.Get());
        ++itWeights;
        ++itIn;
        ++itOut;
      }
    }
  }

  // Make sure that last values are set to 0
  while (!itOut.IsAtEnd())
  {
    pix = itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue(pix);
    itOut.Set(pix);
    ++itOut;
  }
}
} // end namespace rtk
#endif
