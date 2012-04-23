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
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType threadId )
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
        itOut.Set( itIn.Get() + rbiFunctor->GetFarthestDistance() - rbiFunctor->GetNearestDistance() );
      }
    }
}

} // end namespace rtk

#endif
