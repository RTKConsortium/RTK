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

#ifndef rtkDisplacedDetectorForOffsetFieldOfViewImageFilter_hxx
#define rtkDisplacedDetectorForOffsetFieldOfViewImageFilter_hxx

#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkFieldOfViewImageFilter.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
DisplacedDetectorForOffsetFieldOfViewImageFilter<TInputImage, TOutputImage>
::DisplacedDetectorForOffsetFieldOfViewImageFilter():
  m_FOVRadius(-1.),
  m_FOVCenterX(0.),
  m_FOVCenterZ(0.)
{
}

/**
 * When the detector is displaced, one needs to zero pad the input data on the
 * nearest side to the center.
 */
template <class TInputImage, class TOutputImage>
void
DisplacedDetectorForOffsetFieldOfViewImageFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{
  // get pointers to the input and output
  typename Superclass::Superclass::InputImagePointer  inputPtr  = const_cast< TInputImage * >( this->GetInput() );
  typename Superclass::Superclass::OutputImagePointer outputPtr = this->GetOutput();

  if ( !outputPtr || !inputPtr)
    {
    return;
    }

  // Copy the meta data for this data type
  outputPtr->SetSpacing( inputPtr->GetSpacing() );
  outputPtr->SetOrigin( inputPtr->GetOrigin() );
  outputPtr->SetDirection( inputPtr->GetDirection() );
  outputPtr->SetNumberOfComponentsPerPixel( inputPtr->GetNumberOfComponentsPerPixel() );

  typename TOutputImage::RegionType outputLargestPossibleRegion = inputPtr->GetLargestPossibleRegion();

  if(this->GetDisable())
    {
    this->SetInPlace( true );
    outputPtr->SetLargestPossibleRegion( outputLargestPossibleRegion );
    return;
    }
  else if (this->GetGeometry()->GetRadiusCylindricalDetector() != 0)
    {
    itkGenericExceptionMacro(<< "Displaced detector cannot handle cylindrical detector. "
                             << "Consider disabling it by setting m_Disable=true "
                             << "or using the nodisplaced flag of the application you are running");
    }

  typedef typename rtk::FieldOfViewImageFilter<OutputImageType, OutputImageType> FOVFilterType;
  typename FOVFilterType::Pointer fieldofview = FOVFilterType::New();
  fieldofview->SetProjectionsStack( inputPtr.GetPointer() );
  fieldofview->SetGeometry( this->GetGeometry() );
  bool hasOverlap = fieldofview->ComputeFOVRadius(FOVFilterType::RADIUSBOTH, m_FOVCenterX, m_FOVCenterZ, m_FOVRadius);
  double xi, zi, ri;
  fieldofview->ComputeFOVRadius(FOVFilterType::RADIUSINF, xi, zi, ri);
  double xs, zs, rs;
  fieldofview->ComputeFOVRadius(FOVFilterType::RADIUSSUP, xs, zs, rs);

  // 4 cases depending on the position of the two corners
  // Case 1: Impossible to account for too large displacements
  if(!hasOverlap)
    {
    itkGenericExceptionMacro(
          << "Cannot account for too large detector displacements, a part of"
          << " space must be covered by all projections.");
    }
  // Case 2: Not displaced if less than 10% relative difference between radii
  else if( 200. * fabs(ri-rs)/(ri+rs) < 10. )
    {
    this->SetInPlace( true );
    }
  else if( rs > ri )
    {
    this->SetInPlace( false );
    itk::Index<3>::IndexValueType index = outputLargestPossibleRegion.GetIndex()[0] - outputLargestPossibleRegion.GetSize()[0];
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
DisplacedDetectorForOffsetFieldOfViewImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType itkNotUsed(threadId) )
{
  // Compute overlap between input and output
  OutputImageRegionType overlapRegion = outputRegionForThread;

  overlapRegion.Crop(this->GetInput()->GetLargestPossibleRegion() );

  // Input / ouput iterators
  itk::ImageRegionConstIterator<InputImageType> itIn(this->GetInput(), overlapRegion);
  itk::ImageRegionIterator<OutputImageType>     itOut(this->GetOutput(), outputRegionForThread);
  itIn.GoToBegin();
  itOut.GoToBegin();

  // Not displaced, nothing to do
  if( this->GetInput()->GetLargestPossibleRegion().GetSize()[0] ==
      this->GetOutput()->GetLargestPossibleRegion().GetSize()[0] || this->GetDisable())
    {
    // If not in place, copy is required
    if(this->GetInput() != this->GetOutput() )
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
  for(unsigned int k=0; k<overlapRegion.GetSize(2); k++, itWeights.GoToBegin())
    {
    typename GeometryType::HomogeneousVectorType sourcePosition;
    sourcePosition = this->GetGeometry()->GetSourcePosition(itIn.GetIndex()[2]);
    double oppInvNormSource = -1. / (sourcePosition[0]*sourcePosition[0] + sourcePosition[2]*sourcePosition[2]);
    const double sourceToCenterFOVX = m_FOVCenterX - sourcePosition[0];
    const double sourceToCenterFOVZ = m_FOVCenterZ - sourcePosition[2];
    double invNormSourceCenterFOV = 1. / sqrt(sourceToCenterFOVX * sourceToCenterFOVX + sourceToCenterFOVZ * sourceToCenterFOVZ);
    double centerFOVAngle =
        atan2(sourceToCenterFOVZ * invNormSourceCenterFOV, sourceToCenterFOVX * invNormSourceCenterFOV) -
        atan2(sourcePosition[2]  * oppInvNormSource,       sourcePosition[0]  * oppInvNormSource);
    if(centerFOVAngle>vnl_math::pi)  centerFOVAngle -= 2*vnl_math::pi;
    if(centerFOVAngle<-vnl_math::pi) centerFOVAngle += 2*vnl_math::pi;
    double theta2 = asin(m_FOVRadius / sqrt( pow(m_FOVCenterX-sourcePosition[0], 2.) + pow(m_FOVCenterZ-sourcePosition[2], 2.) ));
    double theta1 = -1. * theta2;
    theta1 += centerFOVAngle;
    theta2 += centerFOVAngle;

    // Prepare weights for current slice (depends on ProjectionOffsetsX)
    const double sx  = this->GetGeometry()->GetSourceOffsetsX()[itIn.GetIndex()[2]];
    double sid = this->GetGeometry()->GetSourceToIsocenterDistances()[itIn.GetIndex()[2]];
    sid = sqrt(sid * sid + sx * sx); // To untilted situation
    const double liml1 = tan(theta1) * sid;
    const double liml2 = tan(theta2) * sid;
    double invsid = 0.;
    double piOverDen = 0.;
    if (sid!=0.)
      {
      invsid = 1./sid;
      piOverDen = itk::Math::pi_over_2*sqrt(sourceToCenterFOVX * sourceToCenterFOVX + sourceToCenterFOVZ * sourceToCenterFOVZ)/m_FOVRadius;
      }
    typename WeightImageType::PointType point;
    weights->TransformIndexToPhysicalPoint(itWeights.GetIndex(), point);

    if( this->GetInput()->GetLargestPossibleRegion().GetIndex()[0] !=
        this->GetOutput()->GetLargestPossibleRegion().GetIndex()[0] )
      {
      while(!itWeights.IsAtEnd() )
        {
        const double l = this->GetGeometry()->ToUntiltedCoordinateAtIsocenter(itIn.GetIndex()[2], point[0]);
        if(l <= liml1)
          itWeights.Set(0.);
        else if(l >= liml2)
          itWeights.Set(2.);
        else
          itWeights.Set( sin( sin(atan( l * invsid ) - centerFOVAngle) * piOverDen) + 1.);
        ++itWeights;
        point[0] += spacing[0];
        }
      }
    else
      {
      while(!itWeights.IsAtEnd() )
        {
        const double l = this->GetGeometry()->ToUntiltedCoordinateAtIsocenter(itIn.GetIndex()[2], point[0]);
        if(l <= liml1)
          itWeights.Set(2.0);
        else if(l >= liml2)
          itWeights.Set(0.0);
        else
          itWeights.Set( 1. - sin( sin(atan( l * invsid ) - centerFOVAngle) * piOverDen) );
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
} // end namespace rtk
#endif
