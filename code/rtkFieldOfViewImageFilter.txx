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

#ifndef __rtkFieldOfViewImageFilter_txx
#define __rtkFieldOfViewImageFilter_txx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

namespace rtk
{

template<class TInputImage, class TOutputImage>
FieldOfViewImageFilter<TInputImage, TOutputImage>
::FieldOfViewImageFilter():
  m_Geometry(NULL),
  m_Mask(false)
{
}

template <class TInputImage, class TOutputImage>
void FieldOfViewImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  // Compute projection stack indices of corners
  m_ProjectionsStack->UpdateOutputInformation();
  typename TInputImage::IndexType indexCorner1;
  indexCorner1 = m_ProjectionsStack->GetLargestPossibleRegion().GetIndex();

  typename TInputImage::IndexType indexCorner2;
  indexCorner2 = indexCorner1 + m_ProjectionsStack->GetLargestPossibleRegion().GetSize();
  for(unsigned int i=0; i<TInputImage::GetImageDimension(); i++)
    indexCorner2[i] --;

  // Account for linear interpolation borders
  indexCorner1[0] += 0.5;
  indexCorner1[1] += 0.5;
  indexCorner2[0] -= 0.5;
  indexCorner2[1] -= 0.5;

  // To physical coordinates
  typename TInputImage::PointType corner1, corner2;
  m_ProjectionsStack->TransformIndexToPhysicalPoint(indexCorner1, corner1);
  m_ProjectionsStack->TransformIndexToPhysicalPoint(indexCorner2, corner2);

  // Go over projection stack, compute minimum radius and minimum tangent
  m_Radius = itk::NumericTraits<double>::max();
  m_HatTangent = itk::NumericTraits<double>::max();
  m_HatHeight = itk::NumericTraits<double>::max();
  for(unsigned int k=0; k<m_ProjectionsStack->GetLargestPossibleRegion().GetSize(2); k++)
    {
    const double sid = m_Geometry->GetSourceToIsocenterDistances()[k];
    const double sdd = m_Geometry->GetSourceToDetectorDistances()[k];
    const double mag = sid/sdd;

    const double projOffsetX = m_Geometry->GetProjectionOffsetsX()[k];
    const double sourceOffsetX = m_Geometry->GetSourceOffsetsX()[k];
    m_Radius = std::min( m_Radius, vcl_abs( sourceOffsetX+mag*(corner1[0]+projOffsetX-sourceOffsetX) ) );
    m_Radius = std::min( m_Radius, vcl_abs( sourceOffsetX+mag*(corner2[0]+projOffsetX-sourceOffsetX) ) );

    const double projOffsetY = m_Geometry->GetProjectionOffsetsY()[k];
    const double sourceOffsetY = m_Geometry->GetSourceOffsetsY()[k];
    m_HatHeight = std::min(m_HatHeight, vcl_abs( sourceOffsetY+mag*(corner1[1]+projOffsetY-sourceOffsetY)));
    m_HatHeight = std::min(m_HatHeight, vcl_abs( sourceOffsetY+mag*(corner2[1]+projOffsetY-sourceOffsetY)));
    m_HatTangent = std::min(m_HatTangent, vcl_abs(m_HatHeight/sid));
    }
}

template <class TInputImage, class TOutputImage>
void FieldOfViewImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType itkNotUsed(threadId) )
{
  typename TInputImage::DirectionType d = this->GetInput()->GetDirection();
  if( d[0][0]==1. && d[0][1]==0. && d[0][2]==0. &&
      d[1][0]==0. && d[1][1]==1. && d[1][2]==0. &&
      d[2][0]==0. && d[2][1]==0. && d[2][2]==1.)

    {
    // Prepare point increment (TransformIndexToPhysicalPoint too slow)
    typename TInputImage::PointType pointBase, pointIncrement;
    typename TInputImage::IndexType index = outputRegionForThread.GetIndex();
    this->GetInput()->TransformIndexToPhysicalPoint( index, pointBase );
    for(unsigned int i=0; i<TInputImage::GetImageDimension(); i++)
      index[i]++;
    this->GetInput()->TransformIndexToPhysicalPoint( index, pointIncrement );
    for(unsigned int i=0; i<TInputImage::GetImageDimension(); i++)
      pointIncrement[i] -= pointBase[i];
  
    // Iterators
    typedef itk::ImageRegionConstIterator<TInputImage> InputConstIterator;
    InputConstIterator itIn(this->GetInput(0), outputRegionForThread);
    itIn.GoToBegin();
    typedef itk::ImageRegionIterator<TOutputImage> OutputIterator;
    OutputIterator itOut(this->GetOutput(), outputRegionForThread);
    itOut.GoToBegin();
  
    // Go over output, compute weights and avoid redundant computation
    typename TInputImage::PointType point = pointBase;
    for(unsigned int k=0; k<outputRegionForThread.GetSize(2); k++)
      {
      double zsquare = point[2]*point[2];
      point[1] = pointBase[1];
      for(unsigned int j=0; j<outputRegionForThread.GetSize(1); j++)
        {
        point[0] = pointBase[0];
        for(unsigned int i=0; i<outputRegionForThread.GetSize(0); i++)
          {
          double radius = vcl_sqrt(point[0]*point[0] + zsquare);
          if ( radius <= m_Radius && radius*m_HatTangent<=m_HatHeight-vnl_math_abs(point[1]) )
            {
            if(m_Mask)
              itOut.Set(1.0);
            else
              itOut.Set(itIn.Get());
            }
          else
            itOut.Set(0.);
          ++itIn;
          ++itOut;
          point[0] += pointIncrement[0];
          }
        point[1] += pointIncrement[1];
        }
      point[2] += pointIncrement[2];
      }
    }
  else
    {
    typedef itk::ImageRegionConstIteratorWithIndex<TInputImage> InputConstIterator;
    InputConstIterator itIn(this->GetInput(0), outputRegionForThread);

    typedef itk::ImageRegionIterator<TOutputImage> OutputIterator;
    OutputIterator itOut(this->GetOutput(), outputRegionForThread);

    typename TInputImage::PointType point;
    while( !itIn.IsAtEnd() ) 
      {
      this->GetInput()->TransformIndexToPhysicalPoint( itIn.GetIndex(), point );
      double radius = vcl_sqrt(point[0]*point[0] + point[2]*point[2]);
      if ( radius <= m_Radius && radius*m_HatTangent<=m_HatHeight-vnl_math_abs(point[1]) )
        {
        if(m_Mask)
          itOut.Set(1.0);
        else
          itOut.Set(itIn.Get());
        }
      else
        itOut.Set(0.);
      ++itIn;
      ++itOut;
      }
    }
}

} // end namespace rtk

#endif // __rtkFieldOfViewImageFilter_txx
