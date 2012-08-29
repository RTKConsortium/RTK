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

#ifndef __rtkRayCastInterpolatorForwardProjectionImageFilter_txx
#define __rtkRayCastInterpolatorForwardProjectionImageFilter_txx

#include "rtkHomogeneousMatrix.h"
#include "rtkRayCastInterpolateImageFunction.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
void
RayCastInterpolatorForwardProjectionImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType itkNotUsed(threadId) )
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nPixelPerProj = outputRegionForThread.GetSize(0)*outputRegionForThread.GetSize(1);
  const typename Superclass::GeometryPointer geometry = this->GetGeometry();

  // Create interpolator
  typedef typename rtk::RayCastInterpolateImageFunction< TInputImage, double > InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetThreshold( 0. );
  interpolator->SetInputImage( this->GetInput(1) );
  interpolator->SetTransform(itk::IdentityTransform<double,3>::New());

  // Iterators on volume input and output
  typedef itk::ImageRegionConstIterator<TInputImage> InputRegionIterator;
  InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
  typedef itk::ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
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
  typename Superclass::GeometryType::ThreeDHomogeneousMatrixType volOriginInv;
  volOriginInv = Superclass::GeometryType::
                 ComputeTranslationHomogeneousMatrix(-this->GetInput(1)->GetOrigin()[0],
                                                     -this->GetInput(1)->GetOrigin()[1],
                                                     -this->GetInput(1)->GetOrigin()[2]);

  // Translations are meant to overcome the fact that rtk::RayCastInterpolateFunction
  // forces the origin of the volume at the volume center
  typename Superclass::GeometryType::ThreeDHomogeneousMatrixType volRayCastOrigin;
  volRayCastOrigin.SetIdentity();
  for(unsigned int i=0; i<3; i++)
    volRayCastOrigin[i][3] = -0.5 * this->GetInput(1)->GetSpacing()[i] *
                             (this->GetInput(1)->GetLargestPossibleRegion().GetSize()[i]-1);

  // Combine volume related matrices
  typename Superclass::GeometryType::ThreeDHomogeneousMatrixType volMatrix;
  volMatrix = volRayCastOrigin * volDirectionInv * volOriginInv;

  // Go over each projection
  for(unsigned int iProj=outputRegionForThread.GetIndex(2);
                   iProj<outputRegionForThread.GetIndex(2)+outputRegionForThread.GetSize(2);
                   iProj++)
    {
    // Compute source position and change coordinate system
    typename Superclass::GeometryType::HomogeneousVectorType sourcePosition;
    sourcePosition = volMatrix * geometry->GetSourcePosition(iProj);
    interpolator->SetFocalPoint( &sourcePosition[0] );

    // Compute matrix to transform projection index to volume coordinates
    typename Superclass::GeometryType::ThreeDHomogeneousMatrixType matrix;
    matrix = volMatrix.GetVnlMatrix() *
             geometry->GetProjectionCoordinatesToFixedSystemMatrix(iProj).GetVnlMatrix() *
             GetIndexToPhysicalPointMatrix( this->GetOutput() ).GetVnlMatrix();

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

} // end namespace rtk

#endif
