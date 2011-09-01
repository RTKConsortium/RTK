#ifndef __itkParkerShortScanImageFilter_txx
#define __itkParkerShortScanImageFilter_txx

#include <itkImageRegionIteratorWithIndex.h>
#include <itkMacro.h>

namespace itk
{

template <class TInputImage, class TOutputImage>
void
ParkerShortScanImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId)
{
  // Get angular gaps and max gap
  std::vector<double> angularGaps = m_Geometry->GetAngularGapsWithNext();
  int                 nProj = angularGaps.size();
  int                 maxAngularGapPos = 0;
  for(int iProj=1; iProj<nProj; iProj++)
    if(angularGaps[iProj] > angularGaps[maxAngularGapPos])
      maxAngularGapPos = iProj;

  // Input / ouput iterators
  itk::ImageRegionConstIterator<InputImageType> itIn(this->GetInput(), outputRegionForThread);
  itk::ImageRegionIterator<OutputImageType>     itOut(this->GetOutput(), outputRegionForThread);
  itIn.GoToBegin();
  itOut.GoToBegin();

  // Not a short scan if less than 20 degrees max gap, => nothing to do
  if( angularGaps[maxAngularGapPos] < Math::pi / 9 )
    {
    if(this->GetInput() != this->GetOutput() ) // If not in place, copy is
                                               // required
      {
      while(!itIn.IsAtEnd() )
        {
        itOut.Set( itIn.Get() );
        ++itIn;
        ++itOut;
        }
      }
    return;
    }

  // Weight image parameters
  typename WeightImageType::RegionType region;
  typename WeightImageType::SpacingType spacing;
  typename WeightImageType::PointType origin;
  region.SetSize(0, outputRegionForThread.GetSize(0) );
  region.SetIndex(0, outputRegionForThread.GetIndex(0) );
  spacing[0] = this->GetInput()->GetSpacing()[0];
  origin[0] = this->GetInput()->GetOrigin()[0];

  //Create one line of weights
  typename WeightImageType::Pointer weights = WeightImageType::New();
  weights->SetSpacing( spacing );
  weights->SetOrigin( origin );
  weights->SetRegions( region );
  weights->Allocate();
  typename itk::ImageRegionIteratorWithIndex<WeightImageType> itWeights(weights, weights->GetLargestPossibleRegion() );

  const std::vector<double> rotationAngles = m_Geometry->GetGantryAngles();
  const double detectorWidth = this->GetInput()->GetSpacing()[0] *
                               this->GetInput()->GetLargestPossibleRegion().GetSize()[0];

  // Compute delta angle where there is weighting required
  const double firstAngle = rotationAngles[(maxAngularGapPos+2) % rotationAngles.size()];
  const double lastAngle = rotationAngles[(maxAngularGapPos+rotationAngles.size()-1) % rotationAngles.size()];
  double delta = 0.5 * (lastAngle - firstAngle - 180);
  delta = delta-360*floor(delta/360); // between -360 and 360
  if(delta<0) delta += 360;           // between 0    and 360
  delta *= Math::pi / 180;            // degreees to radians

  double invsdd = 1/m_Geometry->GetSourceToDetectorDistances()[itIn.GetIndex()[2]];
  if( delta < atan(0.5 * detectorWidth * invsdd) )
    itkWarningMacro(<< "You do not have enough data for proper Parker weighting (short scan)");

  for(unsigned int k=0; k<outputRegionForThread.GetSize(2); k++)
    {
    invsdd = 1/m_Geometry->GetSourceToDetectorDistances()[itIn.GetIndex()[2]];

    // Prepare weights for current slice (depends on ProjectionOffsetsX)
    typename WeightImageType::PointType point;
    weights->TransformIndexToPhysicalPoint(itWeights.GetIndex(), point);
    point[0] -= m_Geometry->GetProjectionOffsetsX()[itIn.GetIndex()[2]];

    // Parker's article assumes that the scan starts at 0, convert projection
    // angle accordingly
    double beta = rotationAngles[ itIn.GetIndex()[2] ];
    beta = beta - firstAngle;
    if (beta<0)
      beta += 360;
    beta *= Math::pi / 180;

    itWeights.GoToBegin();
    while(!itWeights.IsAtEnd() )
      {
      double alpha = atan( -1 * point[0] * invsdd );
      if(beta <= 2*delta-2*alpha)
        itWeights.Set( 2. * pow(sin( (Math::pi*beta) / (4*(delta-alpha) ) ), 2.) );
      else if(beta <= Math::pi-2*alpha)
        itWeights.Set( 2. );
      else if(beta <= Math::pi+2*delta)
        itWeights.Set( 2. * pow(sin( (Math::pi*(Math::pi+2*delta-beta) ) / (4*(delta+alpha) ) ), 2.) );
      else
        itWeights.Set( 0. );

      ++itWeights;
      point[0] += spacing[0];
      }

    // Multiply each line of the current slice
    for(unsigned int j=0; j<outputRegionForThread.GetSize(1); j++)
      {
      itWeights.GoToBegin();
      while(!itWeights.IsAtEnd() )
        {
        itOut.Set( itIn.Get() * itWeights.Get() );
        ++itWeights;
        ++itIn;
        ++itOut;
        }
      }
    }
}

} // end namespace itk
#endif
