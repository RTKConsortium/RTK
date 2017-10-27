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

#ifndef rtkFDKWeightProjectionFilter_hxx
#define rtkFDKWeightProjectionFilter_hxx

#include <itkImageRegionIterator.h>

namespace rtk
{
template <class TInputImage, class TOutputImage>
void
FDKWeightProjectionFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  // Get angular weights from geometry
  m_ConstantProjectionFactor = m_Geometry->GetAngularGaps( m_Geometry->GetSourceAngles() );
  m_TiltAngles = m_Geometry->GetTiltAngles();

  for(unsigned int k=0; k<m_ConstantProjectionFactor.size(); k++)
    {
    // Add correction factor for ramp filter
    const double sdd  = m_Geometry->GetSourceToDetectorDistances()[k];
    if(sdd==0.) // Parallel
      m_ConstantProjectionFactor[k] *= 0.5;
    else        // Divergent
      {
      // See [Rit and Clackdoyle, CT meeting, 2014]
      ThreeDCircularProjectionGeometry::HomogeneousVectorType sp;
      sp = m_Geometry->GetSourcePosition(k);
      sp[3] = 0.;
      const double sid  = m_Geometry->GetSourceToIsocenterDistances()[k];
      m_ConstantProjectionFactor[k] *= std::abs(sdd) / (2. * sid * sid);
      m_ConstantProjectionFactor[k] *= sp.GetNorm();
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
    const double sdd  = m_Geometry->GetSourceToDetectorDistances()[k];
    if(sdd != 0.) // Divergent
      {
      typename InputImageType::PointType point = pointBase;
      point[1] = pointBase[1]
                 + m_Geometry->GetProjectionOffsetsY()[k]
                 - m_Geometry->GetSourceOffsetsY()[k];
      const double cosa = cos(m_TiltAngles[k]);
      const double sina = sin(m_TiltAngles[k]);
      const double tana = tan(m_TiltAngles[k]);
      const double sid  = m_Geometry->GetSourceToIsocenterDistances()[k];
      const double sdd2 = sdd * sdd;
      const double RD   = sdd - sid;

      const double numpart1 = sdd*(cosa+tana*sina);
      const double sddtana = sdd * tana;

      for(unsigned int j=0;
                       j<outputRegionForThread.GetSize(1);
                       j++, point[1] += pointIncrement[1])
        {
        point[0] = pointBase[0]
                   + m_Geometry->GetProjectionOffsetsX()[k]
                   + tana * RD;
        const double sdd2y2 = sdd2 + point[1]*point[1];
        for(unsigned int i=0;
                         i<outputRegionForThread.GetSize(0);
                         i++, ++itI, ++itO, point[0] += pointIncrement[0])
          {
          const double denom = sqrt( sdd2y2 + pow(point[0]-sddtana,2.) );
          const double cosGamma = (numpart1 - point[0] * sina) / denom;
          itO.Set( itI.Get() * m_ConstantProjectionFactor[k] * cosGamma );
          }
        }
      }
    else // Parallel
      {
      double weight = m_ConstantProjectionFactor[k];
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
