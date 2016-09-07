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

#ifndef rtkFDKWarpBackProjectionImageFilter_hxx
#define rtkFDKWarpBackProjectionImageFilter_hxx

#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>

#define BILINEAR_BACKPROJECTION

namespace rtk
{

template <class TInputImage, class TOutputImage, class TDeformation>
void
FDKWarpBackProjectionImageFilter<TInputImage,TOutputImage,TDeformation>
::BeforeThreadedGenerateData()
{
  this->SetTranspose(true);
  typename TOutputImage::RegionType splitRegion;
  m_Barrier = itk::Barrier::New();
  m_Barrier->Initialize( this->SplitRequestedRegion(0, this->GetNumberOfThreads(), splitRegion) );
  m_DeformationUpdateError = false;
}

/**
 * GenerateData performs the accumulation
 */
template <class TInputImage, class TOutputImage, class TDeformation>
void
FDKWarpBackProjectionImageFilter<TInputImage,TOutputImage,TDeformation>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId)
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension-1);
  const unsigned int iFirstProj = this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension-1);

  // Create interpolator, could be any interpolation
  typedef itk::LinearInterpolateImageFunction< ProjectionImageType, double > InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

  // Iterators on volume input and output
  typedef itk::ImageRegionConstIterator<TInputImage> InputRegionIterator;
  InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
  typedef itk::ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Initialize output region with input region in case the filter is not in
  // place
  if(this->GetInput() != this->GetOutput() )
    {
    itIn.GoToBegin();
    while(!itIn.IsAtEnd() )
      {
      itOut.Set(itIn.Get() );
      ++itIn;
      ++itOut;
      }
    }

  // Rotation center (assumed to be at 0 yet)
  typename TInputImage::PointType rotCenterPoint;
  rotCenterPoint.Fill(0.0);

  // Continuous index at which we interpolate
  itk::ContinuousIndex<double, Dimension-1> pointProj;

  // Warped point and interpolator for vector field
  typename TOutputImage::PointType point;
  typedef itk::LinearInterpolateImageFunction< typename TDeformation::OutputImageType, double > WarpInterpolatorType;
  typename WarpInterpolatorType::Pointer warpInterpolator = WarpInterpolatorType::New();

  itk::Matrix<double, Dimension+1, Dimension+1> matrixVol =
    GetPhysicalPointToIndexMatrix< TOutputImage >( this->GetOutput() );

  // Go over each projection
  for(unsigned int iProj=iFirstProj; iProj<iFirstProj+nProj; iProj++)
    {
    // Set the deformation
    m_Barrier->Wait();
    if(threadId==0)
      {
      try
        {
        m_Deformation->SetFrame(iProj);
        m_Deformation->Update();
        }
      catch( itk::ExceptionObject & err )
        {
        m_DeformationUpdateError = true;
        m_Barrier->Wait();
        throw err;
        }
      }
    m_Barrier->Wait();
    if(m_DeformationUpdateError)
      return;

    warpInterpolator->SetInputImage(m_Deformation->GetOutput());

    // Extract the current slice
    ProjectionImagePointer projection = this->template GetProjection< ProjectionImageType >(iProj);
    interpolator->SetInputImage(projection);

    // Index to index matrix normalized to have a correct backprojection weight
    // (1 at the isocenter)
    ProjectionMatrixType matrix(this->GetIndexToIndexProjectionMatrix(iProj).GetVnlMatrix() * matrixVol.GetVnlMatrix());
    double perspFactor = matrix[Dimension-1][Dimension];
    for(unsigned int j=0; j<Dimension; j++)
      perspFactor += matrix[Dimension-1][j] * rotCenterPoint[j];
    matrix /= perspFactor;

    // Go over each voxel
    itOut.GoToBegin();
    while(!itOut.IsAtEnd() )
      {
      // Warp
      this->GetOutput()->TransformIndexToPhysicalPoint(itOut.GetIndex(), point);
      if(warpInterpolator->IsInsideBuffer(point))
        point = point + warpInterpolator->Evaluate(point);

      // Compute projection index
      for(unsigned int i=0; i<Dimension-1; i++)
        {
        pointProj[i] = matrix[i][Dimension];
        for(unsigned int j=0; j<Dimension; j++)
          pointProj[i] += matrix[i][j] * point[j];
        }

      // Apply perspective
      double perspFactor = matrix[Dimension-1][Dimension];
      for(unsigned int j=0; j<Dimension; j++)
        perspFactor += matrix[Dimension-1][j] * point[j];
      perspFactor = 1/perspFactor;
      for(unsigned int i=0; i<Dimension-1; i++)
        pointProj[i] = pointProj[i]*perspFactor;

      // Interpolate if in projection
      if( interpolator->IsInsideBuffer(pointProj) )
        {
        itOut.Set( itOut.Get() + perspFactor*perspFactor*interpolator->EvaluateAtContinuousIndex(pointProj) );
        }

      ++itOut;
      }
    }
}

} // end namespace rtk

#endif
