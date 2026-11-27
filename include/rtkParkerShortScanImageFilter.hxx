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

#ifndef rtkParkerShortScanImageFilter_hxx
#define rtkParkerShortScanImageFilter_hxx


#include <itkImageRegionIteratorWithIndex.h>
#include <itkMacro.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
void
ParkerShortScanImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() const
{
  this->Superclass::VerifyPreconditions();

  if (this->GetGeometry() == nullptr)
    itkExceptionMacro(<< "Geometry has not been set.");
}

template <class TInputImage, class TOutputImage>
void
ParkerShortScanImageFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
{
  const ThreeDCircularProjectionGeometry * geometry = this->GetGeometry();
  Superclass::GenerateOutputInformation();

  // Not a short scan if the first and last angles are equal
  const auto [firstAngle, lastAngle] = geometry->GetShortScanFirstLastAngles(geometry->GetGantryAngles());
  m_IsShortScan = firstAngle != lastAngle;
  if (!m_IsShortScan)
    return;
  m_FirstAngle = firstAngle;

  // Delta
  m_Delta = 0.5 * (lastAngle - m_FirstAngle - itk::Math::pi);

  // Check that the short scan covers the detector: pre-compute the two corners of the projection images
  typename TInputImage::IndexType id = this->GetInput()->GetLargestPossibleRegion().GetIndex();
  typename TInputImage::SizeType  sz = this->GetInput()->GetLargestPossibleRegion().GetSize();
  typename TInputImage::SizeType  ones;
  ones.Fill(1);
  typename TInputImage::PointType corner1, corner2;
  this->GetInput()->TransformIndexToPhysicalPoint(id, corner1);
  this->GetInput()->TransformIndexToPhysicalPoint(id - ones + sz, corner2);

  // Go over projection images
  auto lpr = this->GetOutput()->GetLargestPossibleRegion();
  for (unsigned int k = 0; k < lpr.GetSize(2); k++)
  {
    double sox = geometry->GetSourceOffsetsX()[k];
    double sid = geometry->GetSourceToIsocenterDistances()[k];
    double invsid = 1. / sqrt(sid * sid + sox * sox);

    // Check that Parker weighting is relevant for this projection
    double halfDetectorWidth1 = std::abs(geometry->ToUntiltedCoordinateAtIsocenter(k, corner1[0]));
    double halfDetectorWidth2 = std::abs(geometry->ToUntiltedCoordinateAtIsocenter(k, corner2[0]));
    double halfDetectorWidth = std::min(halfDetectorWidth1, halfDetectorWidth2);
    if (m_Delta < atan(halfDetectorWidth * invsid))
    {
      itkWarningMacro(<< "You do not have enough data for proper Parker weighting (short scan)"
                      << " according to projection #" << k << ". Delta is " << m_Delta * 180. / itk::Math::pi
                      << " degrees and should be more than half the beam angle, i.e. "
                      << atan(halfDetectorWidth * invsid) * 180. / itk::Math::pi << " degrees.");
      return;
    }
  }
}

template <class TInputImage, class TOutputImage>
void
ParkerShortScanImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread)
{
  const ThreeDCircularProjectionGeometry * geometry = this->GetGeometry();
  // Input / ouput iterators
  itk::ImageRegionIterator<OutputImageType> itOut(this->GetOutput(), outputRegionForThread);

  // Not a short scan, nothing to do
  if (!m_IsShortScan)
  {
    if (this->GetInput() != this->GetOutput()) // If not in place, copy is
                                               // required
    {
      itk::ImageRegionConstIterator<InputImageType> itIn(this->GetInput(), outputRegionForThread);
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
  region.SetSize(0, outputRegionForThread.GetSize(0));
  region.SetIndex(0, outputRegionForThread.GetIndex(0));
  spacing[0] = this->GetInput()->GetSpacing()[0];
  origin[0] = this->GetInput()->GetOrigin()[0];

  // Create one line of weights
  auto weights = WeightImageType::New();
  weights->SetSpacing(spacing);
  weights->SetOrigin(origin);
  weights->SetRegions(region);
  weights->Allocate();
  typename itk::ImageRegionIteratorWithIndex<WeightImageType> itWeights(weights, weights->GetLargestPossibleRegion());

  const std::vector<double> rotationAngles = geometry->GetGantryAngles();

  // Go over projection images
  itk::ImageRegionConstIteratorWithIndex<InputImageType> itIn(this->GetInput(), outputRegionForThread);
  for (unsigned int k = 0; k < outputRegionForThread.GetSize(2); k++)
  {
    double sox = geometry->GetSourceOffsetsX()[itIn.GetIndex()[2]];
    double sid = geometry->GetSourceToIsocenterDistances()[itIn.GetIndex()[2]];
    double invsid = 1. / sqrt(sid * sid + sox * sox);

    // Prepare weights for current slice (depends on ProjectionOffsetsX)
    typename WeightImageType::PointType point;
    weights->TransformIndexToPhysicalPoint(itWeights.GetIndex(), point);

    // Parker's article assumes that the scan starts at 0, convert projection
    // angle accordingly
    double beta = rotationAngles[itIn.GetIndex()[2]];
    beta = beta - m_FirstAngle;
    if (beta < 0)
      beta += 2 * itk::Math::pi;

    itWeights.GoToBegin();
    while (!itWeights.IsAtEnd())
    {
      const double l = geometry->ToUntiltedCoordinateAtIsocenter(itIn.GetIndex()[2], point[0]);
      double       alpha = atan(-1 * l * invsid);
      const double pi = itk::Math::pi;
      if (beta <= 2 * m_Delta - 2 * alpha)
        itWeights.Set(2. * pow(sin((pi * beta) / (4 * (m_Delta - alpha))), 2.));
      else if (beta <= pi - 2 * alpha)
        itWeights.Set(2.);
      else if (beta <= pi + 2 * m_Delta)
        // Denominator fix to a typo in equation (12) of Parker's article.
        itWeights.Set(2. * pow(sin((pi * (pi + 2 * m_Delta - beta)) / (4 * (m_Delta + alpha))), 2.));
      else
        itWeights.Set(0.);

      ++itWeights;
      point[0] += spacing[0];
    }

    // Multiply each line of the current slice
    for (unsigned int j = 0; j < outputRegionForThread.GetSize(1); j++)
    {
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
}

} // end namespace rtk
#endif
