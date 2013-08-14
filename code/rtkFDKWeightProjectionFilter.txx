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

#ifndef __rtkFDKWeightProjectionFilter_txx
#define __rtkFDKWeightProjectionFilter_txx

#include <itkImageRegionIterator.h>
#include <itkImageFileWriter.h>
#include <itkMeanProjectionImageFilter.h>

namespace rtk
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
    const double sdd  = m_Geometry->GetSourceToDetectorDistances()[k];
    if(sdd==0.) // Parallel
      m_AngularWeightsAndRampFactor[k] *= 0.5;
    else        // Divergent
      {
      // Zoom + factor 1/2 in eq 176, page 106, Kak & Slaney
      const double sid  = m_Geometry->GetSourceToIsocenterDistances()[k];
      const double rampFactor = sdd / (2. * sid);
      m_AngularWeightsAndRampFactor[k] *= rampFactor;
      }
    }
}

template <class TInputImage, class TOutputImage>
void
FDKWeightProjectionFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType itkNotUsed(threadId))
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
  typedef itk::ImageRegionConstIterator<InputImageType> InputConstIterator;
  InputConstIterator itI(this->GetInput(), outputRegionForThread);
  typedef itk::ImageRegionIterator<OutputImageType> OutputIterator;
  OutputIterator itO(this->GetOutput(), outputRegionForThread);
  itI.GoToBegin();
  itO.GoToBegin();

  // Go over output, compute weights and avoid redundant computation
  for(int k=outputRegionForThread.GetIndex(2);
          k<outputRegionForThread.GetIndex(2)+(int)outputRegionForThread.GetSize(2);
          k++)
    {
    typename InputImageType::PointType point = pointBase;
    point[1] = pointBase[1]
               + m_Geometry->GetProjectionOffsetsY()[k]
               - m_Geometry->GetSourceOffsetsY()[k];
    const double sdd  = m_Geometry->GetSourceToDetectorDistances()[k];
    const double sdd2 = sdd * sdd;
    if(sdd != 0.) // Divergent
      {
      const double tauOverD  = m_Geometry->GetSourceOffsetsX()[k] / sdd;
      const double tauOverDw = m_AngularWeightsAndRampFactor[k] * tauOverD;
      const double sddw      = m_AngularWeightsAndRampFactor[k] * sdd;
      for(unsigned int j=0;
                       j<outputRegionForThread.GetSize(1);
                       j++, point[1] += pointIncrement[1])
        {
        point[0] = pointBase[0]
                   + m_Geometry->GetProjectionOffsetsX()[k]
                   - m_Geometry->GetSourceOffsetsX()[k];
        const double sdd2y2 = sdd2 + point[1]*point[1];
        for(unsigned int i=0;
                         i<outputRegionForThread.GetSize(0);
                         i++, ++itI, ++itO, point[0] += pointIncrement[0])
          {
          // The term between parentheses comes from the publication
          // [Gullberg Crawford Tsui, TMI, 1986], equation 18
          itO.Set( itI.Get() * (sddw - tauOverDw * point[0]) / sqrt( sdd2y2 + point[0]*point[0]) );
          }
        }
      }
    else // Parallel
      {
      double weight = m_AngularWeightsAndRampFactor[k];
      for(unsigned int j=0; j<outputRegionForThread.GetSize(1); j++)
        {
        for(unsigned int i=0; i<outputRegionForThread.GetSize(0); i++, ++itI, ++itO)
          itO.Set( itI.Get() * weight);
        }
      }
    }
}

} // end namespace rtk
#endif
