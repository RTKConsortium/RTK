#ifndef __itkForwardProjectionImageFilter_txx
#define __itkForwardProjectionImageFilter_txx

#include "rtkHomogeneousMatrix.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>
#include <itkRayCastInterpolateImageFunction.h>

namespace itk
{

template <class TInputImage, class  TOutputImage>
void
ForwardProjectionImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion()
{
  // Input 0 is the stack of projections in which we project
  typename Superclass::InputImagePointer inputPtr0 =
    const_cast< TInputImage * >( this->GetInput(0) );
  if ( !inputPtr0 )
    return;
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the volume to forward project
  typename Superclass::InputImagePointer  inputPtr1 =
    const_cast< TInputImage * >( this->GetInput(1) );
  if ( !inputPtr1 )
    return;

  typename TInputImage::RegionType reqRegion = inputPtr1->GetLargestPossibleRegion();
  inputPtr1->SetRequestedRegion( reqRegion );
}

template <class TInputImage, class TOutputImage>
void
ForwardProjectionImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType threadId )
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nPixelPerProj = outputRegionForThread.GetSize(0)*outputRegionForThread.GetSize(1);

  // Create interpolator
  typedef typename itk::RayCastInterpolateImageFunction< TInputImage, double > InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetInputImage( this->GetInput(1) );
  interpolator->SetTransform(itk::IdentityTransform<double,3>::New());

  // Iterators on volume input and output
  typedef ImageRegionConstIterator<TInputImage> InputRegionIterator;
  InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
  typedef ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Get inverse volume direction in an homogeneous matrix
  itk::Matrix<double, Dimension, Dimension> volDirInvNotHom;
  volDirInvNotHom = this->GetInput(1)->GetDirection().GetInverse();
  itk::Matrix<double, Dimension+1, Dimension+1> volDirectionInv;
  volDirectionInv.SetIdentity();
  for(unsigned int i=0; i<Dimension; i++)
    for(unsigned int j=0; j<Dimension; j++)
      volDirectionInv[i][j] = volDirInvNotHom[i][j];

  // Get inverse origin in an homogeneous matrix
  itk::Matrix<double, Dimension+1, Dimension+1> volOriginInv;
  volOriginInv = Get3DTranslationHomogeneousMatrix(-this->GetInput(1)->GetOrigin()[0],
                                                   -this->GetInput(1)->GetOrigin()[1],
                                                   -this->GetInput(1)->GetOrigin()[2]);

  // Translations are meant to overcome the fact that itk::RayCastInterpolateFunction
  // forces the origin of the volume at the volume center
  itk::Matrix<double, Dimension+1, Dimension+1> volRayCastOrigin;
  volRayCastOrigin.SetIdentity();
  for(unsigned int i=0; i<3; i++)
    volRayCastOrigin[i][3] = -0.5 * this->GetInput(1)->GetSpacing()[i] *
                             (this->GetInput(1)->GetLargestPossibleRegion().GetSize()[i]-1);

  // Combine volume related matrices
  itk::Matrix<double, Dimension+1, Dimension+1> volMatrix;
  volMatrix = volRayCastOrigin * volDirectionInv * volOriginInv;

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
    rotMatrix = volMatrix * rotMatrix;

    // Compute source position an change coordinate system
    itk::Vector<double, 4> sourcePosition;
    sourcePosition[0] = this->m_Geometry->GetSourceOffsetsX()[iProj];
    sourcePosition[1] = this->m_Geometry->GetSourceOffsetsY()[iProj];
    sourcePosition[2] = this->m_Geometry->GetSourceToIsocenterDistances()[iProj];
    sourcePosition[3] = 1.;
    sourcePosition = rotMatrix * sourcePosition;
    interpolator->SetFocalPoint( typename InterpolatorType::InputPointType(&sourcePosition[0]) );

    // Compute matrix to transform projection index to volume coordinates
    itk::Matrix<double, Dimension+1, Dimension+1> matrix;
    matrix = GetIndexToPhysicalPointMatrix< TOutputImage >( this->GetOutput() );
    matrix[0][3] -= this->m_Geometry->GetProjectionOffsetsX()[iProj] - this->m_Geometry->GetSourceOffsetsX()[iProj];
    matrix[1][3] -= this->m_Geometry->GetProjectionOffsetsX()[iProj] - this->m_Geometry->GetSourceOffsetsY()[iProj];
    matrix[2][3] = this->m_Geometry->GetSourceToIsocenterDistances()[iProj] -
                   this->m_Geometry->GetSourceToDetectorDistances()[iProj];
    matrix[2][2] = 0.; // Force z to axis to detector distance
    matrix = rotMatrix * matrix;

    // Go over each pixel of the projection
    typename TInputImage::PointType point;
    for(unsigned int pix=0; pix<nPixelPerProj; pix++)
      {
      // Compute point coordinate in volume depending on projection index
      for(unsigned int i=0; i<Dimension; i++)
        {
        point[i] = matrix[i][Dimension];
        for(unsigned int j=0; j<Dimension; j++)
          point[i] += matrix[i][j] * itOut.GetIndex()[j];
        }

      itOut.Set( itIn.Get() + interpolator->Evaluate(point) );
      ++itIn;
      ++itOut;
      }
    }
}

} // end namespace itk

#endif
