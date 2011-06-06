#ifndef __itkFDKWeightProjectionFilter_txx
#define __itkFDKWeightProjectionFilter_txx

#include <itkImageRegionIterator.h>
#include <itkImageFileWriter.h>
#include <itkMeanProjectionImageFilter.h>

namespace itk
{
template <class TInputImage, class TOutputImage>
void
FDKWeightProjectionFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  // Get angular weights from geometry
  m_AngularWeightsAndRampFactor = this->GetGeometry()->GetAngularGaps();

  for(unsigned int k=0; k<m_AngularWeightsAndRampFactor.size(); k++)
    {
    // Add correction factor for ramp filter
    const double sid  = m_Geometry->GetSourceToIsocenterDistances()[k];
    const double sdd  = m_Geometry->GetSourceToDetectorDistances()[k];
    // Zoom + factor 1/2 in eq 176, page 106, Kak & Slaney
    const double rampFactor = sdd / (2 * sid);
    m_AngularWeightsAndRampFactor[k] *= rampFactor;
    }
}

template <class TInputImage, class TOutputImage>
void
FDKWeightProjectionFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId)
{
  // Prepare point increment (TransformIndexToPhysicalPoint too slow)
  typename InputImageType::PointType pointBase, pointIncrement;
  typename InputImageType::IndexType index = outputRegionForThread.GetIndex();
  this->GetInput()->TransformIndexToPhysicalPoint( index, pointBase );
  for(int i=0; i<3; i++)
    index[i]++;
  this->GetInput()->TransformIndexToPhysicalPoint( index, pointIncrement );
  for(int i=0; i<3; i++)
    pointIncrement[i] -= pointBase[i];

  // Iterators
  typedef ImageRegionConstIterator<InputImageType> InputConstIterator;
  InputConstIterator itI(this->GetInput(), outputRegionForThread);
  typedef ImageRegionIterator<OutputImageType> OutputIterator;
  OutputIterator itO(this->GetOutput(), outputRegionForThread);
  itI.GoToBegin();
  itO.GoToBegin();

  // Go over output, compute weights and avoid redundant computation
  for(unsigned int k=outputRegionForThread.GetIndex(2);
                   k<outputRegionForThread.GetIndex(2)+outputRegionForThread.GetSize(2);
                   k++)
    {
    typename InputImageType::PointType point = pointBase;
    point[1] = pointBase[1]
               + m_Geometry->GetSourceOffsetsY()[k]
               - m_Geometry->GetProjectionOffsetsY()[k];
    const double sdd  = m_Geometry->GetSourceToDetectorDistances()[k];
    const double sdd2 = sdd * sdd;
    double weight = sdd  * m_AngularWeightsAndRampFactor[k];

    for(unsigned int j=outputRegionForThread.GetIndex(1);
                     j<outputRegionForThread.GetIndex(1)+outputRegionForThread.GetSize(1);
                     j++, point[1] += pointIncrement[1])
      {
      point[0] = pointBase[0]
                 + m_Geometry->GetSourceOffsetsX()[k]
                 - m_Geometry->GetProjectionOffsetsX()[k];
      const double sdd2y2 = sdd2 + point[1]*point[1];
      for(unsigned int i=outputRegionForThread.GetIndex(0);
                       i<outputRegionForThread.GetIndex(0)+outputRegionForThread.GetSize(0);
                       i++, ++itI, ++itO, point[0] += pointIncrement[0])
        {
        itO.Set( itI.Get() * weight / sqrt( sdd2y2 + point[0]*point[0]) );
        }
      }
    }
}

} // end namespace itk
#endif
