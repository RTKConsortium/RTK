#ifndef __itkBackProjectionImageFilter_txx
#define __itkBackProjectionImageFilter_txx

#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>

namespace itk
{

template <class TInputImage, class TOutputImage>
void
BackProjectionImageFilter<TInputImage,TOutputImage>
::GenerateOutputInformation()
{
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput();
  if ( !outputPtr )
    {
    return;
    }

  typename OutputImageRegionType::IndexType index;
  index.Fill(0);

  OutputImageRegionType region;
  region.SetSize(this->m_TomographyDimension);
  region.SetIndex( index );
  outputPtr->SetLargestPossibleRegion( region );

  outputPtr->SetOrigin(this->m_TomographyOrigin);
  outputPtr->SetSpacing(this->m_TomographySpacing);
}


template <class TInputImage, class  TOutputImage>
void
BackProjectionImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer  inputPtr =
    const_cast< TInputImage * >( this->GetInput() );
  if ( !inputPtr )
    {
    return;
    }
  inputPtr->SetRequestedRegion( this->GetInput()->GetLargestPossibleRegion() );
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
  InputImagePointer inputPtr = const_cast< TInputImage * >( this->GetInput() );
  OutputImagePointer outputPtr = this->GetOutput();

  typedef itk::LinearInterpolateImageFunction< TInputImage, double > InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetInputImage( inputPtr );

  // Compute two matrices to go from index to phyisal point and vice-versa
  const unsigned int Dimension = TInputImage::ImageDimension;
  itk::Matrix<double, Dimension+1, Dimension+1> matrixVol  = GetIndexToPhysicalPointMatrix< OutputImageType >(outputPtr);
  itk::Matrix<double, Dimension+1, Dimension+1> matrixTmp = GetPhysicalPointToIndexMatrix< InputImageType >(inputPtr);
  itk::Matrix<double, Dimension, Dimension> matrixProj;
  for(unsigned int i=0; i<Dimension-1; i++)
    {
    for(unsigned int j=0; j<Dimension-1; j++)
      matrixProj[i][j] = matrixTmp[i][j];
    matrixProj[i][Dimension-1] = matrixTmp[i][Dimension];
    }
  matrixProj[Dimension-1][Dimension-1] = 1.0;

  ContinuousIndex<double, Dimension> pointProj;
  for(unsigned int iProj=0; iProj<inputPtr->GetLargestPossibleRegion().GetSize(Dimension-1); iProj+=1+this->m_SkipProjection)
    {
    // Create an index to index projection matrix instead of the physical point 
    // to physical point projection matrix provided by Geometry
    rtk::Geometry<Dimension>::MatrixType matrix = matrixProj.GetVnlMatrix() *
                                                  this->m_Geometry->GetMatrices()[iProj].GetVnlMatrix() *
                                                  matrixVol.GetVnlMatrix();
    typedef ImageRegionIteratorWithIndex<TOutputImage> RegionIterator;
    RegionIterator it(this->GetOutput(), outputRegionForThread);
    while(!it.IsAtEnd())
      {
      pointProj.Fill(0.0);
      for(unsigned int i=0; i<Dimension; i++)
        {
        for(unsigned int j=0; j<Dimension; j++)
          pointProj[i] += matrix[i][j] * it.GetIndex()[j];
        pointProj[i] += matrix[i][Dimension];        
        }

      for(unsigned int i=0; i<Dimension-1; i++)
        pointProj[i] = pointProj[i]/pointProj[Dimension-1];
      pointProj[Dimension-1] = iProj;

      if( interpolator->IsInsideBuffer( pointProj ) )
        it.Set( it.Get() + interpolator->EvaluateAtContinuousIndex( pointProj ) );

      ++it;
      }
    }
}

template <class TInputImage, class TOutputImage>
template <class Args_Info>
void
BackProjectionImageFilter<TInputImage,TOutputImage>
::SetFromGengetopt(const Args_Info & args_info)
{
  const unsigned int Dimension = TOutputImage::ImageDimension;

  this->SetSkipProjection( args_info.skip_proj_arg );

  OutputImageSizeType tomographyDimension;
  tomographyDimension.Fill(args_info.dimension_arg[0]);
  for(unsigned int i=0; i<vnl_math_min(args_info.dimension_given, Dimension); i++)
    tomographyDimension[i] = args_info.dimension_arg[i];
  this->SetTomographyDimension(tomographyDimension);

  OutputImageSpacingType tomographySpacing;
  tomographySpacing.Fill(args_info.spacing_arg[0]);
  for(unsigned int i=0; i<vnl_math_min(args_info.spacing_given, Dimension); i++)
    tomographySpacing[i] = args_info.spacing_arg[i];
  this->SetTomographySpacing(tomographySpacing);

  OutputImagePointType tomographyOrigin;
  for(unsigned int i=0; i<Dimension; i++)
    tomographyOrigin[i] = tomographySpacing[i] * (tomographyDimension[i]-1) * -0.5;
  for(unsigned int i=0; i<vnl_math_min(args_info.origin_given, Dimension); i++)
    tomographyOrigin[i] = args_info.origin_arg[i];
  this->SetTomographyOrigin(tomographyOrigin);
}

} // end namespace itk


#endif
