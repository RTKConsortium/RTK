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

#ifndef __rtkRayBoxIntersectionImageFilter_txx
#define __rtkRayBoxIntersectionImageFilter_txx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include "rtkHomogeneousMatrix.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage,TOutputImage>
::SetBoxFromImage(OutputImageBaseConstPointer _arg)
{
  m_RBIFunctor->SetBoxFromImage(_arg);
  this->Modified();
}

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage,TOutputImage>
::SetBoxMin(VectorType _boxMin)
{
  m_RBIFunctor->SetBoxMin(_boxMin);
  this->Modified();
}

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage,TOutputImage>
::SetBoxMax(VectorType _boxMax)
{
  m_RBIFunctor->SetBoxMax(_boxMax);
  this->Modified();
}

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType itkNotUsed(threadId) )
{
  // Create local object for multithreading purposes
  RBIFunctionType::Pointer rbiFunctor = RBIFunctionType::New();
  rbiFunctor->SetBoxMin(m_RBIFunctor->GetBoxMin());
  rbiFunctor->SetBoxMax(m_RBIFunctor->GetBoxMax());

  // Iterators on input and output
  typedef itk::ImageRegionConstIterator<TInputImage> InputRegionIterator;
  InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
  typedef itk::ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nPixelPerProj = outputRegionForThread.GetSize(0)*outputRegionForThread.GetSize(1);

  // Go over each projection
  for(unsigned int iProj=outputRegionForThread.GetIndex(2);
                   iProj<outputRegionForThread.GetIndex(2)+outputRegionForThread.GetSize(2);
                   iProj++)
    {
    // Set source position
    GeometryType::HomogeneousVectorType sourcePosition = m_Geometry->GetSourcePosition(iProj);
    rbiFunctor->SetRayOrigin( &(sourcePosition[0]) );

    // Compute matrix to transform projection index to volume coordinates
    GeometryType::ThreeDHomogeneousMatrixType matrix;
    matrix = m_Geometry->GetProjectionCoordinatesToFixedSystemMatrix(iProj).GetVnlMatrix() *
             GetIndexToPhysicalPointMatrix( this->GetOutput() ).GetVnlMatrix();

    // Go over each pixel of the projection
    typename RBIFunctionType::VectorType direction;
    for(unsigned int pix=0; pix<nPixelPerProj; pix++, ++itIn, ++itOut)
      {
      // Compute point coordinate in volume depending on projection index
      for(unsigned int i=0; i<Dimension; i++)
        {
        direction[i] = matrix[i][Dimension];
        for(unsigned int j=0; j<Dimension; j++)
          direction[i] += matrix[i][j] * itOut.GetIndex()[j];

        // Direction (projection position - source position)
        direction[i] -= sourcePosition[i];
        }

      // Normalize direction
      double invNorm = 1/direction.GetNorm();
      for(unsigned int i=0; i<Dimension; i++)
        direction[i] *= invNorm;

      // Compute ray intersection length
      if( rbiFunctor->Evaluate(direction) )
        itOut.Set( itIn.Get() + m_Density*(rbiFunctor->GetFarthestDistance() - rbiFunctor->GetNearestDistance()) );
      }
    }
}

} // end namespace rtk

#endif
