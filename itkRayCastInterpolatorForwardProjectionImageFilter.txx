#ifndef __itkRayCastInterpolatorForwardProjectionImageFilter_txx
#define __itkRayCastInterpolatorForwardProjectionImageFilter_txx

#include "rtkHomogeneousMatrix.h"
#include "itkRayCastInterpolateImageFunction.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>

namespace itk
{

template <class TInputImage, class TOutputImage>
void
RayCastInterpolatorForwardProjectionImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType threadId )
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nPixelPerProj = outputRegionForThread.GetSize(0)*outputRegionForThread.GetSize(1);
  const typename Superclass::GeometryType::Pointer geometry = this->GetGeometry();

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
    rotMatrix = Get3DRigidTransformationHomogeneousMatrix( geometry->GetOutOfPlaneAngles()[iProj],
                                                           geometry->GetGantryAngles()[iProj],
                                                           geometry->GetInPlaneAngles()[iProj],
                                                           0.,0.,0.);
    rotMatrix = volMatrix * rotMatrix.GetInverse();

    // Compute source position an change coordinate system
    itk::Vector<double, 4> sourcePosition;
    sourcePosition[0] = geometry->GetSourceOffsetsX()[iProj];
    sourcePosition[1] = geometry->GetSourceOffsetsY()[iProj];
    sourcePosition[2] = -geometry->GetSourceToIsocenterDistances()[iProj];
    sourcePosition[3] = 1.;
    sourcePosition = rotMatrix * sourcePosition;
    interpolator->SetFocalPoint( typename InterpolatorType::InputPointType(&sourcePosition[0]) );

    // Compute matrix to transform projection index to volume coordinates
    itk::Matrix<double, Dimension+1, Dimension+1> matrix;
    matrix = GetIndexToPhysicalPointMatrix< TOutputImage >( this->GetOutput() );
    matrix[0][3] -= geometry->GetProjectionOffsetsX()[iProj] - geometry->GetSourceOffsetsX()[iProj];
    matrix[1][3] -= geometry->GetProjectionOffsetsY()[iProj] - geometry->GetSourceOffsetsY()[iProj];
    matrix[2][3] = geometry->GetSourceToDetectorDistances()[iProj] -
                   geometry->GetSourceToIsocenterDistances()[iProj];
    matrix[2][2] = 0.; // Force z to axis to detector distance
    matrix = rotMatrix * matrix;

    // Go over each pixel of the projection
    typename TInputImage::PointType point;
    for(unsigned int pix=0; pix<nPixelPerProj; pix++, ++itIn, ++itOut)
      {
      // Compute point coordinate in volume depending on projection index
      for(unsigned int i=0; i<Dimension; i++)
        {
        point[i] = matrix[i][Dimension];
        for(unsigned int j=0; j<Dimension; j++)
          point[i] += matrix[i][j] * itOut.GetIndex()[j];
        }

      itOut.Set( itIn.Get() + interpolator->Evaluate(point) );
      }
    }
}

} // end namespace itk

#endif
