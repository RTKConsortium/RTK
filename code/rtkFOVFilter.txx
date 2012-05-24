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

#ifndef __rtkFOVFilter_txx
#define __rtkFOVFilter_txx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include "rtkHomogeneousMatrix.h"
#include "vnl/vnl_math.h"

namespace rtk
{

template<class TInputImage, class TOutputImage>
FOVFilter<TInputImage, TOutputImage>
::FOVFilter():
  m_FOVradius(0.),
  m_ChineseHatHeight(0.),
  m_Geometry(NULL),
  m_Mask(false)
{
  this->SetNumberOfRequiredInputs(2);
}

template <class TInputImage, class TOutputImage>
void FOVFilter<TInputImage, TOutputImage>::ComputationFOVradius()
{
  // Prepare point increment (TransformIndexToPhysicalPoint too slow)
  typename TInputImage::PointType pointBase, pointIncrement, pointExtreme;
  typename TInputImage::IndexType index = this->GetInput(1)->GetLargestPossibleRegion().GetIndex();
  typename TInputImage::IndexType indexExtreme = index + this->GetInput(1)->GetLargestPossibleRegion().GetSize();

  const double sid  = m_Geometry->GetSourceToIsocenterDistances()[0];
  const double sdd  = m_Geometry->GetSourceToDetectorDistances()[0];

  //First pixel in physical mesures, pointBase
  this->GetInput(1)->TransformIndexToPhysicalPoint( index, pointBase );
  typename TInputImage::PointType pointA = pointBase;
  typename TInputImage::PointType pointB;
  typename TInputImage::PointType pointBaseXextreme = pointA;
  typename TInputImage::PointType pointOrigin = pointBase;

  for(int i=0; i<3; i++)
    index[i]++;
  //Calculation of the spacing between each pixel, pointIncrement
  this->GetInput()->TransformIndexToPhysicalPoint( index, pointIncrement );
  for(int i=0; i<3; i++)
    pointIncrement[i] -= pointBase[i];

  //x-axes extreme
  indexExtreme[0] = indexExtreme[0] - 1;
  //Calculation of the extreme location in physical coordinates, pointExtreme
  this->GetInput(1)->TransformIndexToPhysicalPoint( indexExtreme, pointExtreme );
  pointOrigin[0] += (pointExtreme[0] - pointBase[0])/2;

  //We move pointBaseXextreme to the x-axes extreme
  pointBaseXextreme[0] = pointExtreme[0];
  //Auxiliar minimum
  double minAuxiliar = pointExtreme[0];
  // Go over output, compute weights and avoid redundant computation
  for(unsigned int k=this->GetInput(1)->GetLargestPossibleRegion().GetIndex(2);
                   k<this->GetInput(1)->GetLargestPossibleRegion().GetIndex(2)+this->GetInput(1)->GetLargestPossibleRegion().GetSize(2);
                   k++)
  {
    pointA[0] = vcl_abs(pointBase[0] + m_Geometry->GetProjectionOffsetsX()[k] - pointOrigin[0]);
    pointB[0] = vcl_abs(pointBaseXextreme[0] + m_Geometry->GetProjectionOffsetsX()[k] - pointOrigin[0]);
    double minX  = vnl_math_min(pointA[0], pointB[0]);
    minAuxiliar = vnl_math_min(minAuxiliar, minX);
  }
  m_FOVradius = (sid*minAuxiliar)/sdd;
}

template <class TInputImage, class TOutputImage>
void FOVFilter<TInputImage, TOutputImage>::ComputationChineseHatHeight()
{
    // Prepare point increment (TransformIndexToPhysicalPoint too slow)
    typename TInputImage::PointType pointBase, pointIncrement, pointExtreme;
    typename TInputImage::IndexType index = this->GetInput(1)->GetLargestPossibleRegion().GetIndex();
    typename TInputImage::IndexType indexExtreme = index + this->GetInput(1)->GetLargestPossibleRegion().GetSize();

    const double sid  = m_Geometry->GetSourceToIsocenterDistances()[0];
    const double sdd  = m_Geometry->GetSourceToDetectorDistances()[0];

    //First pixel in physical mesures, pointBase
    this->GetInput(1)->TransformIndexToPhysicalPoint( index, pointBase );
    typename TInputImage::PointType pointA = pointBase;
    typename TInputImage::PointType pointB;
    typename TInputImage::PointType pointBaseYextreme = pointA;
    typename TInputImage::PointType pointOrigin = pointBase;

    //y-axes extreme
    indexExtreme[1] = indexExtreme[1] - 1;
    //Calculation of the extreme location in physical coordinates, pointExtreme
    this->GetInput()->TransformIndexToPhysicalPoint( indexExtreme, pointExtreme );
    pointOrigin[1] += (pointExtreme[1] - pointBase[1])/2;

    for(int i=0; i<3; i++)
      index[i]++;
    //Calculation of the spacing between each pixel, pointIncrement
    this->GetInput()->TransformIndexToPhysicalPoint( index, pointIncrement );
    for(int i=0; i<3; i++)
      pointIncrement[i] -= pointBase[i];

    //We move pointBaseYextreme to the y-axes extreme
    pointBaseYextreme[1] = pointExtreme[1];
    //Auxiliar minimum
    double minAuxiliar = pointExtreme[1];
    // Go over output, compute weights and avoid redundant computation
    for(unsigned int k=index[2];
                     k<index[2]+this->GetInput(1)->GetLargestPossibleRegion().GetSize(2);
                     k++)
    {
      pointA[1] = vcl_abs(pointBase[1] + m_Geometry->GetProjectionOffsetsY()[k] - pointOrigin[1]);
      pointB[1] = vcl_abs(pointBaseYextreme[1] + m_Geometry->GetProjectionOffsetsY()[k] - pointOrigin[1]);
      double minX  = vnl_math_min(pointA[1], pointB[1]);
      minAuxiliar = vnl_math_min(minAuxiliar, minX);
    }
    m_ChineseHatHeight = (sid*minAuxiliar)/sdd;
}

template <class TInputImage, class TOutputImage>
void FOVFilter<TInputImage, TOutputImage>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                                                             ThreadIdType threadId )
{
  // Prepare point increment (TransformIndexToPhysicalPoint too slow)
  typename TInputImage::PointType pointBase, pointIncrement;
  typename TInputImage::IndexType index = outputRegionForThread.GetIndex();
  this->GetInput()->TransformIndexToPhysicalPoint( index, pointBase );
  typename TInputImage::PointType point = pointBase;
  for(int i=0; i<3; i++)
    index[i]++;
  this->GetInput()->TransformIndexToPhysicalPoint( index, pointIncrement );
  for(int i=0; i<3; i++)
    pointIncrement[i] -= pointBase[i];

  // Iterators
  typedef itk::ImageRegionConstIterator<TInputImage> InputConstIterator;
  InputConstIterator itIn(this->GetInput(0), outputRegionForThread);
  typedef itk::ImageRegionIterator<TOutputImage> OutputIterator;
  OutputIterator itOut(this->GetOutput(), outputRegionForThread);
  itIn.GoToBegin();
  itOut.GoToBegin();

  this->ComputationFOVradius();
  this->ComputationChineseHatHeight();
  const double sid  = m_Geometry->GetSourceToIsocenterDistances()[0];
  const double teta = vcl_atan(m_ChineseHatHeight/sid);
  double y_min = vcl_tan(teta)*(sid-vnl_math_abs(point[2]));

  // Go over output, compute weights and avoid redundant computation
  for(unsigned int k=outputRegionForThread.GetIndex(2);
                   k<outputRegionForThread.GetIndex(2)+outputRegionForThread.GetSize(2);
                   k++)
  {
    point[1] = pointBase[1];
    point[2] += pointIncrement[2];
    for(unsigned int j=outputRegionForThread.GetIndex(1);
                     j<outputRegionForThread.GetIndex(1)+outputRegionForThread.GetSize(1);
                     j++)
    {
      point[0] = pointBase[0];
      point[1] += pointIncrement[1];
      for(unsigned int i=outputRegionForThread.GetIndex(0);
                       i<outputRegionForThread.GetIndex(0)+outputRegionForThread.GetSize(0);
                       i++, ++itIn, ++itOut)
      {
        double radius = vcl_sqrt(point[0]*point[0] + point[2]*point[2]);
        //Checking if the voxel is inside radius "FOVradius"
        if ( radius <= m_FOVradius )
          {
            y_min = vcl_tan(teta)*(sid-radius);
            //Checking if the voxel is inside the range [ChineseHat, -ChineseHat]
            if(vnl_math_abs(point[1])<=y_min)
              {
              if(m_Mask)
                itOut.Set(1.0);
              else
                itOut.Set(itIn.Get());
              }
          }
        point[0] += pointIncrement[0];
      }
    }
  }
}

}// end namespace rtk

#endif
