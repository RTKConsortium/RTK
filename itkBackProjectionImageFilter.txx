#ifndef __itkBackProjectionImageFilter_txx
#define __itkBackProjectionImageFilter_txx

#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkExtractImageFilter.h>

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
  inputPtr1->SetRequestedRegion( inputPtr1->GetLargestPossibleRegion() );
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

  // The input is a stack of projections, we need to extract only one projection
  // for efficiency during interpolation
  typedef itk::Image<TInputImage::PixelType, Dimension-1> ProjectionImageType;
  typedef itk::ExtractImageFilter< TInputImage, ProjectionImageType > ExtractFilterType;
  ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();
  extractFilter->SetInput(this->GetInput(1));
  ExtractFilterType::InputImageRegionType region = this->GetInput(1)->GetLargestPossibleRegion();
  const unsigned int nProj = region.GetSize(Dimension-1);
  region.SetSize(Dimension-1, 0);
  extractFilter->SetExtractionRegion(region);
  extractFilter->Update();

  // Create interpolator, could be any interpolation
  typedef itk::LinearInterpolateImageFunction< ProjectionImageType, double > InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetInputImage( extractFilter->GetOutput() );

  // Compute two matrices to go from index to phyisal point and vice-versa
  itk::Matrix<double, Dimension+1, Dimension+1> matrixVol  = GetIndexToPhysicalPointMatrix< OutputImageType >(this->GetOutput());
  itk::Matrix<double, Dimension, Dimension> matrixProj = GetPhysicalPointToIndexMatrix< ProjectionImageType >(extractFilter->GetOutput());

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
    region.SetIndex(Dimension-1, iProj);
    extractFilter->SetExtractionRegion(region);
    extractFilter->Update();

    // Create an index to index projection matrix instead of the physical point 
    // to physical point projection matrix provided by Geometry
    rtk::Geometry<Dimension>::MatrixType matrix = matrixProj.GetVnlMatrix() *
                                                  this->m_Geometry->GetMatrices()[iProj].GetVnlMatrix() *
                                                  matrixVol.GetVnlMatrix();

    // Go over each voxel
    itIn.GoToBegin();
    itOut.GoToBegin();
    while(!itIn.IsAtEnd())
      {
      // Compute projection index
      for(unsigned int i=0; i<Dimension-1; i++)
        {
        pointProj[i] = matrix[i][Dimension];
        for(unsigned int j=0; j<Dimension; j++)
          pointProj[i] += matrix[i][j] * itOut.GetIndex()[j];
        }
      double perspFactor = matrix[Dimension-1][Dimension];
      for(unsigned int j=0; j<Dimension; j++)
        perspFactor += matrix[Dimension-1][j] * itOut.GetIndex()[j];
      perspFactor = 1/perspFactor;
      for(unsigned int i=0; i<Dimension-1; i++)
        pointProj[i] = pointProj[i]*perspFactor;

      // Interpolate if in projection
      if( interpolator->IsInsideBuffer(pointProj) )
        itOut.Set( itIn.Get() + interpolator->EvaluateAtContinuousIndex(pointProj) );

      ++itIn;
      ++itOut;
      }
    }
}

} // end namespace itk


#endif
