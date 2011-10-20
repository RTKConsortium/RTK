#ifndef __itkRayBoxIntersectionImageFilter_txx
#define __itkRayBoxIntersectionImageFilter_txx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

namespace itk
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
  typedef ImageRegionConstIterator<TInputImage> InputRegionIterator;
  InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
  typedef ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nPixelPerProj = outputRegionForThread.GetSize(0)*outputRegionForThread.GetSize(1);

  // Go over each projection
  for(unsigned int iProj=outputRegionForThread.GetIndex(2);
                   iProj<outputRegionForThread.GetIndex(2)+outputRegionForThread.GetSize(2);
                   iProj++)
    {
    // Account for system rotations
    itk::Matrix<double, Dimension+1, Dimension+1> rotMatrix;
    rotMatrix = Get3DRigidTransformationHomogeneousMatrix( m_Geometry->GetOutOfPlaneAngles()[iProj],
                                                           m_Geometry->GetGantryAngles()[iProj],
                                                           m_Geometry->GetInPlaneAngles()[iProj],
                                                           0.,0.,0.);
    rotMatrix = rotMatrix.GetInverse();

    // Compute source position an change coordinate system
    itk::Vector<double, 4> sourcePosition;
    sourcePosition[0] = this->m_Geometry->GetSourceOffsetsX()[iProj];
    sourcePosition[1] = this->m_Geometry->GetSourceOffsetsY()[iProj];
    sourcePosition[2] = -this->m_Geometry->GetSourceToIsocenterDistances()[iProj];
    sourcePosition[3] = 1.;
    sourcePosition = rotMatrix * sourcePosition;
    RBIFunctionType::VectorType p;
    p[0] = sourcePosition[0];
    p[1] = sourcePosition[1];
    p[2] = sourcePosition[2];
    rbiFunctor->SetRayOrigin( p );

    // Compute matrix to transform projection index to volume coordinates
    itk::Matrix<double, Dimension+1, Dimension+1> matrix;
    matrix = GetIndexToPhysicalPointMatrix< TOutputImage >( this->GetOutput() );
    matrix[0][3] -= this->m_Geometry->GetProjectionOffsetsX()[iProj] - this->m_Geometry->GetSourceOffsetsX()[iProj];
    matrix[1][3] -= this->m_Geometry->GetProjectionOffsetsY()[iProj] - this->m_Geometry->GetSourceOffsetsY()[iProj];
    matrix[2][3] = this->m_Geometry->GetSourceToDetectorDistances()[iProj] -
                   this->m_Geometry->GetSourceToIsocenterDistances()[iProj];
    matrix[2][2] = 0.; // Force z to axis to detector distance
    matrix = rotMatrix * matrix;

    // Go over each pixel of the projection
    typename RBIFunctionType::VectorType direction;
    for(unsigned int pix=0; pix<nPixelPerProj; pix++)
      {
      // Compute point coordinate in volume depending on projection index
      for(unsigned int i=0; i<Dimension; i++)
        {
        direction[i] = matrix[i][Dimension];
        for(unsigned int j=0; j<Dimension; j++)
          direction[i] += matrix[i][j] * itOut.GetIndex()[j];

        // Direction
        direction[i] -= sourcePosition[i];
        }
      double invNorm = 1/direction.GetNorm();
      for(unsigned int i=0; i<Dimension; i++)
        direction[i] *= invNorm;
      if( rbiFunctor->Evaluate(direction) )
        itOut.Set( itIn.Get() + rbiFunctor->GetFarthestDistance() - rbiFunctor->GetNearestDistance() );
      ++itIn;
      ++itOut;
      }
    }
}

} // end namespace itk

#endif
