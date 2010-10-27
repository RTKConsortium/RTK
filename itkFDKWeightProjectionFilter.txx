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
    region.SetIndex(i, 0);
    region.SetSize(i, this->GetOutput()->GetLargestPossibleRegion().GetSize()[i]);
    }
  this->GetOutput()->SetRequestedRegion(region);
}

template <class TInputImage, class TOutputImage>
void
FDKWeightProjectionFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  // Compute weights image parameters (== one projection, i.e. one slice of the input)
  typename WeightImageType::SizeType size;
  typename WeightImageType::SpacingType spacing;
  typename WeightImageType::PointType origin;
  for(unsigned int i=0; i<WeightImageType::ImageDimension; i++){
    size[i] = this->GetInput()->GetLargestPossibleRegion().GetSize()[i];
    spacing[i] = this->GetInput()->GetSpacing()[i];
    origin[i] = this->GetInput()->GetOrigin()[i];
  }

  // Test if necessary to recompute the weights image
  if(m_WeightsImage.GetPointer() != NULL && 
     m_WeightsImage->GetSpacing() == spacing &&
     m_WeightsImage->GetOrigin() == origin &&
     m_WeightsImage->GetLargestPossibleRegion().GetSize() == size)
     return;

  // Allocate weights image
  m_WeightsImage = WeightImageType::New();
  m_WeightsImage->SetSpacing( spacing );
  m_WeightsImage->SetOrigin( origin );
  m_WeightsImage->SetRegions( size );
  m_WeightsImage->Allocate();

  // Compute weights
  typedef ImageRegionIteratorWithIndex<WeightImageType> RegionIterator;
  RegionIterator it(m_WeightsImage, m_WeightsImage->GetLargestPossibleRegion());
  double sdd2 = this->m_SourceToDetectorDistance * this->m_SourceToDetectorDistance;
  typename WeightImageType::PointType point;
  while(!it.IsAtEnd())
    {
    m_WeightsImage->TransformIndexToPhysicalPoint( it.GetIndex(), point );
    double sourceToPointDistance = sdd2;
    for(unsigned int i=0; i<WeightImageType::ImageDimension; i++)
      sourceToPointDistance += point[i] * point[i];
    it.Set( this->m_SourceToDetectorDistance / sqrt( sourceToPointDistance ) );
    ++it;
    }
}

template <class TInputImage, class TOutputImage>
void
FDKWeightProjectionFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId)
{
  // Prepare iterators
  typedef ImageRegionConstIterator<InputImageType> InputConstIterator;
  InputConstIterator itI(this->GetInput(), outputRegionForThread);

  typedef ImageRegionConstIterator<WeightImageType> WeightRegionConstIterator;
  typename WeightImageType::RegionType wr = m_WeightsImage->GetLargestPossibleRegion();
  for(unsigned int i=0; i<WeightImageType::ImageDimension; i++)
    {
    wr.SetIndex(i, outputRegionForThread.GetIndex(i));
    wr.SetSize(i, outputRegionForThread.GetSize(i));
    }
  WeightRegionConstIterator itW(m_WeightsImage, wr);

  typedef ImageRegionIterator<OutputImageType> OutputIterator;
  OutputIterator itO(this->GetOutput(), outputRegionForThread);

  // Multiply slice-by-slice
  itI.GoToBegin();
  itO.GoToBegin();
  while(!itI.IsAtEnd()) {
    itW.GoToBegin();
    while(!itW.IsAtEnd()) {
      itO.Set(itI.Get() * itW.Get());
      ++itI;
      ++itW;
      ++itO;
    }
  }
}

} // end namespace itk
#endif
