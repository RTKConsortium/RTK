#ifndef __itkFDKWeightProjectionFilter_txx
#define __itkFDKWeightProjectionFilter_txx

#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageFileWriter.h>

namespace itk
{
template <class TInputImage, class TOutputImage>
void
FDKWeightProjectionFilter<TInputImage, TOutputImage>
::EnlargeOutputRequestedRegion( DataObject *output )
{
  OutputImageRegionType region = this->GetOutput()->GetRequestedRegion();

  for(unsigned int i=0; i<WeightImageType::ImageDimension; i++)
    {
    region.SetIndex(i, this->GetOutput()->GetLargestPossibleRegion().GetIndex()[i]);
    region.SetSize(i, this->GetOutput()->GetLargestPossibleRegion().GetSize()[i]);
    }
  this->GetOutput()->SetRequestedRegion(region);
}

template <class TInputImage, class TOutputImage>
void
FDKWeightProjectionFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  // Compute weights image parameters (== one projection, i.e. one slice of the
  // input)
  typename WeightImageType::RegionType region;
  typename WeightImageType::SpacingType spacing;
  typename WeightImageType::PointType origin;
  for(unsigned int i=0; i<WeightImageType::ImageDimension; i++)
    {
    region.SetSize(i, this->GetInput()->GetLargestPossibleRegion().GetSize()[i]);
    region.SetIndex(i, this->GetInput()->GetLargestPossibleRegion().GetIndex()[i]);
    spacing[i] = this->GetInput()->GetSpacing()[i];
    origin[i] = this->GetInput()->GetOrigin()[i];
    }

  // Test if necessary to recompute the weights image
  if(m_WeightsImage.GetPointer() != NULL &&
     m_WeightsImage->GetSpacing() == spacing &&
     m_WeightsImage->GetOrigin() == origin &&
     m_WeightsImage->GetLargestPossibleRegion() == region)
    return;

  // Allocate weights image
  m_WeightsImage = WeightImageType::New();
  m_WeightsImage->SetSpacing( spacing );
  m_WeightsImage->SetOrigin( origin );
  m_WeightsImage->SetRegions( region );
  m_WeightsImage->Allocate();

  // Correction factor for ramp filter
  double rampFactor = m_Geometry->GetSourceToDetectorDistance() / m_Geometry->GetSourceToIsocenterDistance();
  rampFactor *= 0.5; // Factor 1/2 in eq 176, page 106, Kak & Slaney

  // Compute weights
  typedef ImageRegionIteratorWithIndex<WeightImageType> RegionIterator;
  RegionIterator it(m_WeightsImage, m_WeightsImage->GetLargestPossibleRegion() );
  double sdd2 = pow( m_Geometry->GetSourceToDetectorDistance(), 2);
  typename WeightImageType::PointType point;
  while(!it.IsAtEnd() )
    {
    m_WeightsImage->TransformIndexToPhysicalPoint( it.GetIndex(), point );
    double sourceToPointDistance = sdd2;
    for(unsigned int i=0; i<WeightImageType::ImageDimension; i++)
      sourceToPointDistance += point[i] * point[i];
    it.Set( rampFactor * m_Geometry->GetSourceToDetectorDistance() / sqrt( sourceToPointDistance ) );
    ++it;
    }

  // Get angular weights from geometry
  m_AngularWeights = this->GetGeometry()->GetAngularGaps();
}

template <class TInputImage, class TOutputImage>
void
FDKWeightProjectionFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId)
{
  // Prepare iterators
  typedef ImageRegionConstIteratorWithIndex<InputImageType> InputConstIterator;
  InputConstIterator itI(this->GetInput(), outputRegionForThread);

  typedef ImageRegionConstIterator<WeightImageType> WeightRegionConstIterator;
  typename WeightImageType::RegionType wr = m_WeightsImage->GetLargestPossibleRegion();
  for(unsigned int i=0; i<WeightImageType::ImageDimension; i++)
    {
    wr.SetIndex(i, outputRegionForThread.GetIndex(i) );
    wr.SetSize(i, outputRegionForThread.GetSize(i) );
    }
  WeightRegionConstIterator itW(m_WeightsImage, wr);

  typedef ImageRegionIterator<OutputImageType> OutputIterator;
  OutputIterator itO(this->GetOutput(), outputRegionForThread);

  // Multiply slice-by-slice
  itI.GoToBegin();
  itO.GoToBegin();
  while(!itI.IsAtEnd() )
    {
    itW.GoToBegin();
    while(!itW.IsAtEnd() )
      {
      itO.Set(itI.Get() * itW.Get() * m_AngularWeights[ itI.GetIndex()[2] ] );
      ++itI;
      ++itW;
      ++itO;
      }
    }
}

} // end namespace itk
#endif
