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

#ifndef rtkFDKBackProjectionImageFilter_hxx
#define rtkFDKBackProjectionImageFilter_hxx

#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>

#define BILINEAR_BACKPROJECTION

namespace rtk
{

template <class TInputImage, class TOutputImage>
void
FDKBackProjectionImageFilter<TInputImage,TOutputImage>
::GenerateOutputInformation()
{
  // Check if detector is cylindrical
  if(this->m_Geometry->GetRadiusCylindricalDetector() != 0)
    {
    itkGenericExceptionMacro(<< "Voxel-based FDK back projector can currently not handle cylindrical detectors")
    }

  // Run superclass' GenerateOutputInformation
  Superclass::GenerateOutputInformation();
}

/**
 * GenerateData performs the accumulation
 */
template <class TInputImage, class TOutputImage>
void
FDKBackProjectionImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType itkNotUsed(threadId) )
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
  itk::ContinuousIndex<double, Dimension> rotCenterIndex;
  this->GetInput(0)->TransformPhysicalPointToContinuousIndex(rotCenterPoint, rotCenterIndex);

  // Continuous index at which we interpolate
  itk::ContinuousIndex<double, Dimension-1> pointProj;

  // Go over each projection
  for(unsigned int iProj=iFirstProj; iProj<iFirstProj+nProj; iProj++)
    {
    // Extract the current slice
    ProjectionImagePointer projection;
    projection = this->template GetProjection< ProjectionImageType >(iProj);
    interpolator->SetInputImage(projection);

    // Index to index matrix normalized to have a correct backprojection weight
    // (1 at the isocenter)
    ProjectionMatrixType matrix = this->GetIndexToIndexProjectionMatrix(iProj);
    double perspFactor = matrix[Dimension-1][Dimension];
    for(unsigned int j=0; j<Dimension; j++)
      perspFactor += matrix[Dimension-1][j] * rotCenterIndex[j];
    matrix /= perspFactor;

    // Optimized version
    if (fabs(matrix[1][0])<1e-10 && fabs(matrix[2][0])<1e-10)
      {
      OptimizedBackprojectionX( outputRegionForThread, matrix, projection);
      continue;
      }
    if (fabs(matrix[1][1])<1e-10 && fabs(matrix[2][1])<1e-10)
      {
      OptimizedBackprojectionY( outputRegionForThread, matrix, projection);
      continue;
      }

    // Go over each voxel
    itOut.GoToBegin();
    while(!itOut.IsAtEnd() )
      {
      // Compute projection index
      for(unsigned int i=0; i<Dimension-1; i++)
        {
        pointProj[i] = matrix[i][Dimension];
        for(unsigned int j=0; j<Dimension; j++)
          pointProj[i] += matrix[i][j] * itOut.GetIndex()[j];
        }

      // Apply perspective
      double perspFactor = matrix[Dimension-1][Dimension];
      for(unsigned int j=0; j<Dimension; j++)
        perspFactor += matrix[Dimension-1][j] * itOut.GetIndex()[j];
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

template <class TInputImage, class TOutputImage>
void
FDKBackProjectionImageFilter<TInputImage,TOutputImage>
::OptimizedBackprojectionX(const OutputImageRegionType& region, const ProjectionMatrixType& matrix,
                           const ProjectionImagePointer projection)
{
  typename ProjectionImageType::SizeType pSize = projection->GetBufferedRegion().GetSize();
  typename ProjectionImageType::IndexType pIndex = projection->GetBufferedRegion().GetIndex();
  typename TOutputImage::SizeType vBufferSize = this->GetOutput()->GetBufferedRegion().GetSize();
  typename TOutputImage::IndexType vBufferIndex = this->GetOutput()->GetBufferedRegion().GetIndex();
  typename TInputImage::PixelType *pProj;
  typename TOutputImage::PixelType *pVol, *pVolZeroPointer;

  // Pointers in memory to index (0,0,0) which do not necessarily exist
  pVolZeroPointer = this->GetOutput()->GetBufferPointer();
  pVolZeroPointer -= vBufferIndex[0] + vBufferSize[0] * (vBufferIndex[1] + vBufferSize[1] * vBufferIndex[2]);

  // Continuous index at which we interpolate
  double u, v, w;
  int    ui, vi;
  double du;

  for(int k=region.GetIndex(2); k<region.GetIndex(2)+(int)region.GetSize(2); k++)
    {
    for(int j=region.GetIndex(1); j<region.GetIndex(1)+(int)region.GetSize(1); j++)
      {
      int i = region.GetIndex(0);
      u = matrix[0][0] * i + matrix[0][1] * j + matrix[0][2] * k + matrix[0][3];
      v =                    matrix[1][1] * j + matrix[1][2] * k + matrix[1][3];
      w =                    matrix[2][1] * j + matrix[2][2] * k + matrix[2][3];

      //Apply perspective
      w = 1/w;
      u = u*w-pIndex[0];
      v = v*w-pIndex[1];
      du = w * matrix[0][0];
      w *= w;

#ifdef BILINEAR_BACKPROJECTION
      double u1, u2, v1, v2;
      vi = vnl_math_floor(v);
      if(vi>=0 && vi<(int)pSize[1]-1)
        {
        v1 = v-vi;
        v2 = 1.0-v1;
#else
      vi = itk::Math::Round<double>(v)-pIndex[1];
      if(vi>=0 && vi<(int)pSize[1])
        {
#endif

        pProj = projection->GetBufferPointer() + vi * pSize[0];
        pVol = pVolZeroPointer + i + vBufferSize[0] * (j + k * vBufferSize[1] );

        // Innermost loop
        for(; i<(region.GetIndex(0) + (int)region.GetSize(0)); i++, u += du, pVol++)
          {
#ifdef BILINEAR_BACKPROJECTION
          ui = vnl_math_floor(u);
          if(ui>=0 && ui<(int)pSize[0]-1)
            {
            u1 = u-ui;
            u2 = 1.0-u1;
            *pVol += w * (v2 * (u2 * *(pProj+ui)          + u1 * *(pProj+ui+1) ) +
                          v1 * (u2 * *(pProj+ui+pSize[0]) + u1 * *(pProj+ui+pSize[0]+1) ) );
            }
#else
          ui = itk::Math::Round<double>(u);
          if(ui>=0 && ui<(int)pSize[0])
            {
            *pVol += w * *(pProj+ui);
            }
#endif
          } //i
        }
      } //j
    } //k
}

template <class TInputImage, class TOutputImage>
void
FDKBackProjectionImageFilter<TInputImage,TOutputImage>
::OptimizedBackprojectionY(const OutputImageRegionType& region, const ProjectionMatrixType& matrix,
                           const ProjectionImagePointer projection)
{
  typename ProjectionImageType::SizeType pSize = projection->GetBufferedRegion().GetSize();
  typename ProjectionImageType::IndexType pIndex = projection->GetBufferedRegion().GetIndex();
  typename TOutputImage::SizeType vBufferSize = this->GetOutput()->GetBufferedRegion().GetSize();
  typename TOutputImage::IndexType vBufferIndex = this->GetOutput()->GetBufferedRegion().GetIndex();
  typename TInputImage::PixelType *pProj;
  typename TOutputImage::PixelType *pVol, *pVolZeroPointer;

  // Pointers in memory to index (0,0,0) which do not necessarily exist
  pVolZeroPointer = this->GetOutput()->GetBufferPointer();
  pVolZeroPointer -= vBufferIndex[0] + vBufferSize[0] * (vBufferIndex[1] + vBufferSize[1] * vBufferIndex[2]);

  // Continuous index at which we interpolate
  double u, v, w;
  int    ui, vi;
  double du;

  for(int k=region.GetIndex(2); k<region.GetIndex(2)+(int)region.GetSize(2); k++)
    {
    for(int i=region.GetIndex(0); i<region.GetIndex(0)+(int)region.GetSize(0); i++)
      {
      int j = region.GetIndex(1);
      u = matrix[0][0] * i + matrix[0][1] * j + matrix[0][2] * k + matrix[0][3];
      v = matrix[1][0] * i +                    matrix[1][2] * k + matrix[1][3];
      w = matrix[2][0] * i +                    matrix[2][2] * k + matrix[2][3];

      //Apply perspective
      w = 1/w;
      u = u*w-pIndex[0];
      v = v*w-pIndex[1];
      du = w * matrix[0][1];
      w *= w;

#ifdef BILINEAR_BACKPROJECTION
      vi = vnl_math_floor(v);
      if(vi>=0 && vi<(int)pSize[1]-1)
        {
#else
      vi = itk::Math::Round<double>(v);
      if(vi>=0 && vi<(int)pSize[1])
        {
#endif
        pVol = pVolZeroPointer + i + vBufferSize[0] * (j + k * vBufferSize[1] );
        for(; j<(region.GetIndex(1) + (int)region.GetSize(1)); j++, pVol += vBufferSize[0], u += du)
          {
#ifdef BILINEAR_BACKPROJECTION
          ui = vnl_math_floor(u);
          if(ui>=0 && ui<(int)pSize[0]-1)
            {
            double u1, u2, v1, v2;
            pProj = projection->GetBufferPointer() + vi * pSize[0] + ui;
            v1 = v-vi;
            v2 = 1.0-v1;
            u1 = u-ui;
            u2 = 1.0-u1;
            *pVol += w * (v2 * (u2 * *(pProj)          + u1 * *(pProj+1) ) +
                          v1 * (u2 * *(pProj+pSize[0]) + u1 * *(pProj+pSize[0]+1) ) );
            }
#else
          ui = itk::Math::Round<double>(u);
          if(ui>=0 && ui<(int)pSize[0])
            {
            pProj = projection->GetBufferPointer() + vi * pSize[0];
            *pVol += w * *(pProj+ui);
            }
#endif
          } //j
        }
      } //i
    } //k
}

} // end namespace rtk

#endif
