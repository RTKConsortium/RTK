#ifndef __itkDisplacedDetectorImageFilter_txx
#define __itkDisplacedDetectorImageFilter_txx

#include <itkImageRegionIteratorWithIndex.h>

namespace itk
{

/**
 * Account for the padding computed in GenerateOutputInformation to propagate the
 * requested region.
 */
template <class TInputImage, class TOutputImage>
void
DisplacedDetectorImageFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer  inputPtr = const_cast< TInputImage * >( this->GetInput() );
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput();

  if ( !inputPtr || !outputPtr )
    return;

  typename TInputImage::RegionType inputRequestedRegion = outputPtr->GetRequestedRegion();
  inputRequestedRegion.Crop( inputPtr->GetLargestPossibleRegion() );

  inputPtr->SetRequestedRegion( inputRequestedRegion );
}

/**
 * When the detector is displaced, one needs to zero pad the input data on the
 * nearest side to the center.
 */
template <class TInputImage, class TOutputImage>
void
DisplacedDetectorImageFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{
  // call the superclass' implementation of this method
  Superclass::GenerateOutputInformation();

  // get pointers to the input and output
  typename Superclass::InputImagePointer  inputPtr  = const_cast< TInputImage * >( this->GetInput() );
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();

  if ( !outputPtr || !inputPtr)
    {
    return;
    }

  typename TOutputImage::RegionType outputLargestPossibleRegion = inputPtr->GetLargestPossibleRegion();

  // Compute the X coordinates of the corners of the image
  m_InferiorCorner = inputPtr->GetOrigin()[0] +  inputPtr->GetSpacing()[0] * outputLargestPossibleRegion.GetIndex(0);
  m_SuperiorCorner = m_InferiorCorner;
  if (inputPtr->GetSpacing()[0]<0.)
    m_InferiorCorner += inputPtr->GetSpacing()[0] * (outputLargestPossibleRegion.GetSize(0)-1);
  else
    m_SuperiorCorner += inputPtr->GetSpacing()[0] * (outputLargestPossibleRegion.GetSize(0)-1);

  // Account for projections offsets
  double minOffset = NumericTraits<double>::max();
  double maxOffset = NumericTraits<double>::min();
  for(unsigned int i=0; i<m_Geometry->GetProjectionOffsetsX().size(); i++)
    {
    minOffset = vnl_math_min(minOffset, m_Geometry->GetProjectionOffsetsX()[i]);
    maxOffset = vnl_math_max(maxOffset, m_Geometry->GetProjectionOffsetsX()[i]);
    }
  m_InferiorCorner -= minOffset;
  m_SuperiorCorner -= maxOffset;

  // 4 cases depending on the position of the two corners
  // Case 1: Impossible to account for too large displacements
  if(m_InferiorCorner>0. || m_SuperiorCorner<0.)
    {
    itkGenericExceptionMacro(<< "Can not account for detector displacement larger than 50% of panel size.");
    }
  // Case 2: Not dispaced, nothing to do
  else if( fabs(m_InferiorCorner+m_SuperiorCorner) < 0.1*fabs(m_SuperiorCorner-m_InferiorCorner) )
    {
    this->SetInPlace( true );
    }
  else if( m_SuperiorCorner+m_InferiorCorner > 0. )
    {
    this->SetInPlace( false );
    Index<3>::IndexValueType index = outputLargestPossibleRegion.GetIndex()[0] - outputLargestPossibleRegion.GetSize()[0];
    outputLargestPossibleRegion.SetIndex( 0, index );
    outputLargestPossibleRegion.SetSize( 0, outputLargestPossibleRegion.GetSize()[0]*2 );
    }
  else
    {
    this->SetInPlace( false );
    outputLargestPossibleRegion.SetSize( 0, outputLargestPossibleRegion.GetSize()[0]*2 );
    }

  outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );
}

template <class TInputImage, class TOutputImage>
void
DisplacedDetectorImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId)
{
  // Compute overlap between input and output
  OutputImageRegionType overlapRegion = outputRegionForThread;

  overlapRegion.Crop(this->GetInput()->GetLargestPossibleRegion() );

  // Input / ouput iterators
  itk::ImageRegionConstIterator<InputImageType> itIn(this->GetInput(), overlapRegion);
  itk::ImageRegionIterator<OutputImageType>     itOut(this->GetOutput(), outputRegionForThread);
  itIn.GoToBegin();
  itOut.GoToBegin();

  // Not dispaced, nothing to do
  if( fabs(m_InferiorCorner+m_SuperiorCorner) < 0.1*fabs(m_SuperiorCorner-m_InferiorCorner) )
    {
    if(this->GetInput() != this->GetOutput() ) // If not in place, copy is
                                               // required
      {
      while(!itIn.IsAtEnd() )
        {
        itOut.Set( itIn.Get() );
        ++itIn;
        ++itOut;
        }
      }
    return;
    }

  // Weight image parameters
  typename WeightImageType::RegionType region;
  typename WeightImageType::SpacingType spacing;
  typename WeightImageType::PointType origin;
  region.SetSize(0, overlapRegion.GetSize(0) );
  region.SetIndex(0, overlapRegion.GetIndex(0) );
  spacing[0] = this->GetInput()->GetSpacing()[0];
  origin[0] = this->GetInput()->GetOrigin()[0];

  //Create one line of weights
  typename WeightImageType::Pointer weights = WeightImageType::New();
  weights->SetSpacing( spacing );
  weights->SetOrigin( origin );
  weights->SetRegions( region );
  weights->Allocate();
  typename itk::ImageRegionIteratorWithIndex<WeightImageType> itWeights(weights, weights->GetLargestPossibleRegion() );

  double       theta = vnl_math_min(-1*m_InferiorCorner, m_SuperiorCorner);

  for(unsigned int k=0; k<overlapRegion.GetSize(2); k++)
    {
    // Prepare weights for current slice (depends on ProjectionOffsetsX)
    const double invsdd = 1/m_Geometry->GetSourceToDetectorDistances()[itIn.GetIndex()[2]];
    const double invden = 1/(2 * vcl_atan( theta * invsdd ) );
    typename WeightImageType::PointType point;
    weights->TransformIndexToPhysicalPoint(itWeights.GetIndex(), point);
    point[0] -= m_Geometry->GetProjectionOffsetsX()[itIn.GetIndex()[2]];

    if( m_SuperiorCorner+m_InferiorCorner > 0. )
      {
      while(!itWeights.IsAtEnd() )
        {
        if(point[0] <= -1*theta)
          itWeights.Set(0.0);
        else if(point[0] >= theta)
          itWeights.Set(2.0);
        else
          itWeights.Set( sin( Math::pi*atan(point[0] * invsdd ) * invden ) + 1 );
        ++itWeights;
        point[0] += spacing[0];
        }
      }
    else
      {
      while(!itWeights.IsAtEnd() )
        {
        if(point[0] <= -1*theta)
          itWeights.Set(2.0);
        else if(point[0] >= theta)
          itWeights.Set(0.0);
        else
          itWeights.Set( 1 - sin( Math::pi*atan(point[0] * invsdd ) * invden ) );
        ++itWeights;
        point[0] += spacing[0];
        }
      }

    // Multiply each line of the current slice
    for(unsigned int j=0; j<overlapRegion.GetSize(1); j++)
      {
      // Set outside of overlap to 0 values
      while( itOut.GetIndex()[0] != itIn.GetIndex()[0] )
        {
        itOut.Set( 0 );
        ++itOut;
        }

      itWeights.GoToBegin();
      while(!itWeights.IsAtEnd() )
        {
        itOut.Set( itIn.Get() * itWeights.Get() );
        ++itWeights;
        ++itIn;
        ++itOut;
        }
      }
    }

  // Make sure that last values are set to 0
  while( !itOut.IsAtEnd() )
    {
    itOut.Set( 0 );
    ++itOut;
    }
}

} // end namespace itk
#endif
