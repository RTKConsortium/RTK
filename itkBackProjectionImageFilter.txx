#ifndef __itkBackProjectionImageFilter_txx
#define __itkBackProjectionImageFilter_txx

#include "rtkHomogeneousMatrix.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include "rtkMacro.h"
namespace itk
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
  if(this->GetUpdateProjectionPerProjection() )
    reqRegion.SetSize( TInputImage::ImageDimension-1, 1);
  inputPtr1->SetRequestedRegion( reqRegion );
}

/**
 * GenerateData performs the accumulation
 */
template <class TInputImage, class TOutputImage>
void
BackProjectionImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       int threadId )
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension-1);

  // Create interpolator, could be any interpolation
  typedef itk::LinearInterpolateImageFunction< ProjectionImageType, double > InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

  // Iterators on volume input and output
  typedef ImageRegionConstIterator<TInputImage> InputRegionIterator;
  InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
  typedef ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Continuous index at which we interpolate
  ContinuousIndex<double, Dimension-1> pointProj;

  // Go over each projection
  for(unsigned int iProj=0; iProj<nProj; iProj++)
    {
    // Extract the current slice
    ProjectionImagePointer projection = GetProjection(iProj);
    ProjectionMatrixType   matrix = GetIndexToIndexProjectionMatrix(iProj, projection);
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
        if (iProj)
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

  if(this->GetUpdateProjectionPerProjection() )
    {
    // Lock the projection stack to avoid multithreading issues. Unlocked at the
    // end of this function.
    m_ProjectionStackLock->Lock();

    typename TInputImage::RegionType buffRegion = stack->GetLargestPossibleRegion();
    buffRegion.SetIndex(ProjectionImageType::ImageDimension, iProj);
    buffRegion.SetSize(ProjectionImageType::ImageDimension, 1);

    stack->SetRequestedRegion(buffRegion);
    stack->Update();
    }
  const int iProjBuff = stack->GetBufferedRegion().GetIndex(ProjectionImageType::ImageDimension);

  ProjectionImagePointer projection = ProjectionImageType::New();
  typename ProjectionImageType::RegionType region;
  typename ProjectionImageType::SpacingType spacing;
  typename ProjectionImageType::PointType origin;

  for(unsigned int i=0; i<ProjectionImageType::ImageDimension; i++)
    {
    origin[i] = stack->GetOrigin()[i];
    spacing[i] = stack->GetSpacing()[i];
    region.SetSize(i, stack->GetLargestPossibleRegion().GetSize()[i]);
    region.SetIndex(i, stack->GetLargestPossibleRegion().GetIndex()[i]);
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

  const unsigned int    npixels = projection->GetLargestPossibleRegion().GetNumberOfPixels();
  const InputPixelType *pi = stack->GetBufferPointer() + (iProj-iProjBuff)*npixels;
  InputPixelType *      po = projection->GetBufferPointer();

  // Transpose projection for optimization
  if(this->GetTranspose() )
    {
    for(unsigned int j=0; j<region.GetSize(0); j++, po-=npixels-1)
      for(unsigned int i=0; i<region.GetSize(1); i++, po+=region.GetSize(0) )
        *po = *pi++;
    }
  else
    for(unsigned int i=0; i<npixels; i++)
      *po++ = *pi++;

  if(this->GetUpdateProjectionPerProjection() )
    m_ProjectionStackLock->Unlock();
  return projection;
}

template <class TInputImage, class TOutputImage>
typename BackProjectionImageFilter<TInputImage,TOutputImage>::ProjectionMatrixType
BackProjectionImageFilter<TInputImage,TOutputImage>
::GetIndexToIndexProjectionMatrix(const unsigned int iProj, const ProjectionImageType *proj)
{
  const unsigned int Dimension = TInputImage::ImageDimension;

  itk::Matrix<double, Dimension+1, Dimension+1> matrixVol = GetIndexToPhysicalPointMatrix< TOutputImage >(
      this->GetOutput() );
  itk::Matrix<double, Dimension, Dimension> matrixProj = GetPhysicalPointToIndexMatrix< ProjectionImageType >(proj);

  // Transpose projection for optimization
  if(this->GetTranspose() )
    for(unsigned int i=0; i<Dimension; i++)
      std::swap(matrixProj[i][0], matrixProj[i][1]);

  return ProjectionMatrixType(matrixProj.GetVnlMatrix() *
                              this->m_Geometry->GetMatrices()[iProj].GetVnlMatrix() *
                              matrixVol.GetVnlMatrix() );
}

} // end namespace itk

#endif
