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
 * GenerateData Performs the accumulation
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

  InputImagePointType point2D;
  OutputImagePointType point3D;

  for(unsigned int iProj=0; iProj<inputPtr->GetLargestPossibleRegion().GetSize(2); iProj+=1+this->m_SkipProjection)
    {
    typedef ImageRegionIteratorWithIndex<TOutputImage> RegionIterator;
    RegionIterator it(this->GetOutput(), outputRegionForThread);
    while(!it.IsAtEnd())
      {
      outputPtr->TransformIndexToPhysicalPoint( it.GetIndex(), point3D );

      point2D.Fill(0.0);
      for(unsigned int i=0; i<3; i++)
        {
        for(unsigned int j=0; j<3; j++)
          point2D[i] += this->m_Geometry->GetMatrices()[iProj][i][j] * point3D[j];
        point2D[i] += this->m_Geometry->GetMatrices()[iProj][i][3];        
        }

      point2D[0] = point2D[0]/point2D[2];
      point2D[1] = point2D[1]/point2D[2];
      point2D[2] = iProj;

      if( interpolator->IsInsideBuffer( point2D ) )
        it.Set( it.Get() + interpolator->Evaluate( point2D ) );

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
