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

#ifndef __rtkBackProjectionImageFilter_txx
#define __rtkBackProjectionImageFilter_txx

#include "rtkHomogeneousMatrix.h"
#include "rtkMacro.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>

namespace rtk
{

template <class TInputImage, class  TOutputImage>
void
BackProjectionImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion()
{
  // Input 0 is the volume in which we backproject
  typename Superclass::InputImagePointer inputPtr0 =
    const_cast< TInputImage * >( this->GetInput(0) );
  if ( !inputPtr0 )
    return;
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the stack of projections to backproject
  typename Superclass::InputImagePointer  inputPtr1 =
    const_cast< TInputImage * >( this->GetInput(1) );
  if ( !inputPtr1 )
    return;

  typename TInputImage::RegionType reqRegion = inputPtr1->GetLargestPossibleRegion();
  if(m_Geometry.GetPointer() == NULL)
    {
    inputPtr1->SetRequestedRegion( inputPtr1->GetLargestPossibleRegion() );
    return;
    }

  const unsigned int Dimension = TInputImage::ImageDimension;

  itk::ContinuousIndex<double, Dimension> cornerInf;
  itk::ContinuousIndex<double, Dimension> cornerSup;
  cornerInf[0] = itk::NumericTraits<double>::max();
  cornerSup[0] = itk::NumericTraits<double>::NonpositiveMin();
  cornerInf[1] = itk::NumericTraits<double>::max();
  cornerSup[1] = itk::NumericTraits<double>::NonpositiveMin();
  cornerInf[2] = reqRegion.GetIndex(2);
  cornerSup[2] = reqRegion.GetIndex(2) + reqRegion.GetSize(2);

  // Go over each projection
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension-1);
  const unsigned int iFirstProj = this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension-1);
  this->SetTranspose(false);
  for(int iProj=iFirstProj; iProj<iFirstProj+nProj; iProj++)
    {
    // Extract the current slice
    ProjectionMatrixType   matrix = GetIndexToIndexProjectionMatrix(iProj);

    // Check which part of the projection image will be backprojected in the
    // volume.
    double firstPerspFactor;
    for(int cz=0; cz<2; cz++)
      for(int cy=0; cy<2; cy++)
        for(int cx=0; cx<2; cx++)
          {
          // Compute projection index
          typename TInputImage::IndexType index = this->GetInput()->GetRequestedRegion().GetIndex();
          index[0] += cx*this->GetInput()->GetRequestedRegion().GetSize(0);
          index[1] += cy*this->GetInput()->GetRequestedRegion().GetSize(1);
          index[2] += cz*this->GetInput()->GetRequestedRegion().GetSize(2);

          itk::ContinuousIndex<double, Dimension-1> point;
          for(unsigned int i=0; i<Dimension-1; i++)
            {
            point[i] = matrix[i][Dimension];
            for(unsigned int j=0; j<Dimension; j++)
              point[i] += matrix[i][j] * index[j];
            }

          // Apply perspective
          double perspFactor = matrix[Dimension-1][Dimension];
          for(unsigned int j=0; j<Dimension; j++)
            perspFactor += matrix[Dimension-1][j] * index[j];
          perspFactor = 1/perspFactor;
          for(unsigned int i=0; i<Dimension-1; i++)
            point[i] = point[i]*perspFactor;

          // Check if corners all have the same perspective factor sign.
          // If not, source is too close for easily computing a smaller requested
          // region than the largest possible one.
          if(cx+cy+cz==0)
            firstPerspFactor = perspFactor;
          if(perspFactor*firstPerspFactor < 0.) // Change of sign
            {
            inputPtr1->SetRequestedRegion( inputPtr1->GetLargestPossibleRegion() );
            return;
            }

          // Look for extremas on projection to calculate requested region
          for(int i=0; i<2; i++)
            {
            cornerInf[i] = vnl_math_min(cornerInf[i], point[i]);
            cornerSup[i] = vnl_math_max(cornerSup[i], point[i]);
            }
          }
    }
  reqRegion.SetIndex(0, vnl_math_floor(cornerInf[0]) );
  reqRegion.SetIndex(1, vnl_math_floor(cornerInf[1]) );
  reqRegion.SetSize(0, vnl_math_ceil(cornerSup[0]+1.)-vnl_math_floor(cornerInf[0]) );
  reqRegion.SetSize(1, vnl_math_ceil(cornerSup[1]+1.)-vnl_math_floor(cornerInf[1]) );

  if( reqRegion.Crop( inputPtr1->GetLargestPossibleRegion() ) )
    inputPtr1->SetRequestedRegion( reqRegion );
  else
    {
    inputPtr1->SetRequestedRegion( inputPtr1->GetLargestPossibleRegion() );
    }
}

/**
 * GenerateData performs the accumulation
 */
template <class TInputImage, class TOutputImage>
void
BackProjectionImageFilter<TInputImage,TOutputImage>
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

  // Continuous index at which we interpolate
  itk::ContinuousIndex<double, Dimension-1> pointProj;

  // Go over each projection
  for(unsigned int iProj=iFirstProj; iProj<iFirstProj+nProj; iProj++)
    {
    // Extract the current slice
    ProjectionImagePointer projection = GetProjection(iProj);

    ProjectionMatrixType   matrix = GetIndexToIndexProjectionMatrix(iProj);
    interpolator->SetInputImage(projection);

    // Go over each voxel
    itIn.GoToBegin();
    itOut.GoToBegin();
    while(!itIn.IsAtEnd() )
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
        if (iProj!=iFirstProj)
          itOut.Set( itOut.Get() + interpolator->EvaluateAtContinuousIndex(pointProj) );
        else
          itOut.Set( itIn.Get() + interpolator->EvaluateAtContinuousIndex(pointProj) );
        }

      ++itIn;
      ++itOut;
      }
    }
}

template <class TInputImage, class TOutputImage>
typename BackProjectionImageFilter<TInputImage,TOutputImage>::ProjectionImagePointer
BackProjectionImageFilter<TInputImage,TOutputImage>
::GetProjection(const unsigned int iProj)
{

  typename Superclass::InputImagePointer stack = const_cast< TInputImage * >( this->GetInput(1) );

  const int iProjBuff = stack->GetBufferedRegion().GetIndex(ProjectionImageType::ImageDimension);

  ProjectionImagePointer projection = ProjectionImageType::New();
  typename ProjectionImageType::RegionType region;
  typename ProjectionImageType::SpacingType spacing;
  typename ProjectionImageType::PointType origin;

  for(unsigned int i=0; i<ProjectionImageType::ImageDimension; i++)
    {
    origin[i] = stack->GetOrigin()[i];
    spacing[i] = stack->GetSpacing()[i];
    region.SetSize(i, stack->GetBufferedRegion().GetSize()[i]);
    region.SetIndex(i, stack->GetBufferedRegion().GetIndex()[i]);
    }
  if(this->GetTranspose() )
    {
    typename ProjectionImageType::SizeType size = region.GetSize();
    typename ProjectionImageType::IndexType index = region.GetIndex();
    std::swap(size[0], size[1]);
    std::swap(index[0], index[1]);
    std::swap(origin[0], origin[1]);
    std::swap(spacing[0], spacing[1]);
    region.SetSize(size);
    region.SetIndex(index);
    }
  projection->SetSpacing(spacing);
  projection->SetOrigin(origin);
  projection->SetRegions(region);
  projection->Allocate();

  const unsigned int    npixels = projection->GetBufferedRegion().GetNumberOfPixels();
  const InputPixelType *pi = stack->GetBufferPointer() + (iProj-iProjBuff)*npixels;
  InputPixelType *      po = projection->GetBufferPointer();

  // Transpose projection for optimization
  if(this->GetTranspose() )
    {
    for(unsigned int j=0; j<region.GetSize(0); j++, po -= npixels-1)
      for(unsigned int i=0; i<region.GetSize(1); i++, po += region.GetSize(0))
        *po = *pi++;
    }
  else
    for(unsigned int i=0; i<npixels; i++)
      *po++ = *pi++;

  return projection;
}

template <class TInputImage, class TOutputImage>
typename BackProjectionImageFilter<TInputImage,TOutputImage>::ProjectionMatrixType
BackProjectionImageFilter<TInputImage,TOutputImage>
::GetIndexToIndexProjectionMatrix(const unsigned int iProj)
{
  const unsigned int Dimension = TInputImage::ImageDimension;

  itk::Matrix<double, Dimension+1, Dimension+1> matrixVol =
    GetIndexToPhysicalPointMatrix< TOutputImage >( this->GetOutput() );

  itk::Matrix<double, Dimension+1, Dimension+1> matrixStackProj =
    GetPhysicalPointToIndexMatrix< TOutputImage >( this->GetInput(1) );

  itk::Matrix<double, Dimension, Dimension> matrixProj;
  matrixProj.SetIdentity();
  for(unsigned int i=0; i<Dimension-1; i++)
    {
    matrixProj[i][Dimension-1] = matrixStackProj[i][Dimension];
    for(unsigned int j=0; j<Dimension-1; j++)
      matrixProj[i][j] = matrixStackProj[i][j];
    }

  // Transpose projection for optimization
  itk::Matrix<double, Dimension, Dimension> matrixFlip;
  matrixFlip.SetIdentity();
  if(this->GetTranspose() )
    {
    std::swap(matrixFlip[0][0], matrixFlip[0][1]);
    std::swap(matrixFlip[1][0], matrixFlip[1][1]);
    }

  return ProjectionMatrixType(matrixFlip.GetVnlMatrix() *
                              matrixProj.GetVnlMatrix() *
                              this->m_Geometry->GetMatrices()[iProj].GetVnlMatrix() *
                              matrixVol.GetVnlMatrix() );
}

} // end namespace rtk

#endif
