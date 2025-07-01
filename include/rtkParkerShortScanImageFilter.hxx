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


#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkMacro.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
ParkerShortScanImageFilter<TInputImage, TOutputImage>::ParkerShortScanImageFilter()
  : m_AngularGapThreshold(itk::Math::pi / 9)
{
  this->SetInPlace(true);
}

template <class TInputImage, class TOutputImage>
void
ParkerShortScanImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() const
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull())
    itkExceptionMacro(<< "Geometry has not been set.");
}

template <class TInputImage, class TOutputImage>
void
ParkerShortScanImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();

  // Get angular gaps and max gap
  std::vector<double> angularGaps = m_Geometry->GetAngularGapsWithNext(m_Geometry->GetGantryAngles());
  int                 nProj = angularGaps.size();
  int                 maxAngularGapPos = 0;
  for (int iProj = 1; iProj < nProj; iProj++)
    if (angularGaps[iProj] > angularGaps[maxAngularGapPos])
      maxAngularGapPos = iProj;

  // Not a short scan if less than m_AngularGapThreshold degrees max gap, => nothing to do
  // FIXME: do nothing in parallel geometry, currently handled with a trick in the geometry object
  if (m_Geometry->GetSourceToDetectorDistances()[0] == 0. || angularGaps[maxAngularGapPos] < m_AngularGapThreshold)
  {
    m_IsShortScan = false;
    return;
  }
  m_IsShortScan = true;

  const std::vector<double>            rotationAngles = m_Geometry->GetGantryAngles();
  const std::map<double, unsigned int> sortedAngles = m_Geometry->GetUniqueSortedAngles(m_Geometry->GetGantryAngles());

  // Compute delta between first and last angle where there is weighting required
  std::map<double, unsigned int>::const_iterator itLastAngle;
  itLastAngle = sortedAngles.find(rotationAngles[maxAngularGapPos]);
  auto itFirstAngle = itLastAngle;
  itFirstAngle = (++itFirstAngle == sortedAngles.end()) ? sortedAngles.begin() : itFirstAngle;
  m_FirstAngle = itFirstAngle->first;
  double lastAngle = itLastAngle->first;
  if (lastAngle < m_FirstAngle)
  {
    lastAngle += 2 * itk::Math::pi;
  }
  // Delta
  m_Delta = 0.5 * (lastAngle - m_FirstAngle - itk::Math::pi);
  m_Delta = m_Delta - 2 * itk::Math::pi * floor(m_Delta / (2 * itk::Math::pi)); // between -2*PI and 2*PI

  // Pre-compute the two corners of the projection images
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
    double sox = m_Geometry->GetSourceOffsetsX()[k];
    double sid = m_Geometry->GetSourceToIsocenterDistances()[k];
    double invsid = 1. / sqrt(sid * sid + sox * sox);

    // Check that Parker weighting is relevant for this projection
    double halfDetectorWidth1 = itk::Math::abs(m_Geometry->ToUntiltedCoordinateAtIsocenter(k, corner1[0]));
    double halfDetectorWidth2 = itk::Math::abs(m_Geometry->ToUntiltedCoordinateAtIsocenter(k, corner2[0]));
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
  // Input / ouput iterators
  itk::ImageRegionConstIterator<InputImageType> itIn(this->GetInput(), outputRegionForThread);
  itk::ImageRegionIterator<OutputImageType>     itOut(this->GetOutput(), outputRegionForThread);
  itIn.GoToBegin();
  itOut.GoToBegin();

  // Not a short scan if less than m_AngularGapThreshold degrees max gap, => nothing to do
  // FIXME: do nothing in parallel geometry, currently handled with a trick in the geometry object
  if (!m_IsShortScan)
  {
    if (this->GetInput() != this->GetOutput()) // If not in place, copy is
                                               // required
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

  const std::vector<double> rotationAngles = m_Geometry->GetGantryAngles();

  // Go over projection images
  for (unsigned int k = 0; k < outputRegionForThread.GetSize(2); k++)
  {
    double sox = m_Geometry->GetSourceOffsetsX()[itIn.GetIndex()[2]];
    double sid = m_Geometry->GetSourceToIsocenterDistances()[itIn.GetIndex()[2]];
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
      const double l = m_Geometry->ToUntiltedCoordinateAtIsocenter(itIn.GetIndex()[2], point[0]);
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
