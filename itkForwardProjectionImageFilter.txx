#ifndef __itkForwardProjectionImageFilter_txx
#define __itkForwardProjectionImageFilter_txx

#include "rtkHomogeneousMatrix.h"
#include "itkRayCastInterpolateImageFunction.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>

namespace itk
{

template <class TInputImage, class  TOutputImage>
void
ForwardProjectionImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion()
{
  // Input 0 is the stack of projections in which we project
  typename Superclass::InputImagePointer inputPtr0 =
    const_cast< TInputImage * >( this->GetInput(0) );
  if ( !inputPtr0 )
    return;
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the volume to forward project
  typename Superclass::InputImagePointer  inputPtr1 =
    const_cast< TInputImage * >( this->GetInput(1) );
  if ( !inputPtr1 )
    return;

  typename TInputImage::RegionType reqRegion = inputPtr1->GetLargestPossibleRegion();
  inputPtr1->SetRequestedRegion( reqRegion );
}

} // end namespace itk

#endif
