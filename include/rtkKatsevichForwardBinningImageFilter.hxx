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

#ifndef rtkKatsevichForwardBinningImageFilter_hxx
#define rtkKatsevichForwardBinningImageFilter_hxx

#include "math.h"

#include <rtkHomogeneousMatrix.h>

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkPixelTraits.h>

#include <rtkKatsevichForwardBinningImageFilter.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
void
KatsevichForwardBinningImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  //  // Input 0 is Output ie result of forward binning
  typename Superclass::InputImagePointer inputPtr0 = const_cast<TInputImage *>(this->GetInput());
  if (!inputPtr0)
    return;
  inputPtr0->SetRequestedRegion(this->GetInput()->GetLargestPossibleRegion());
}
// Not necessary as GetInput()->GetLargest = inputPtr0->GetRequested before assignment.

template <typename TInputImage, typename TOutputImage>
void
KatsevichForwardBinningImageFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
{
  // Call the superclass' implementation of this method
  Superclass::GenerateOutputInformation();

  // Get pointers to the input and output
  const InputImageType * inputPtr = this->GetInput();
  OutputImageType *      outputPtr = this->GetOutput();

  itkAssertInDebugAndIgnoreInReleaseMacro(inputPtr);
  itkAssertInDebugAndIgnoreInReleaseMacro(outputPtr != nullptr);

  unsigned int                              i;
  const typename TInputImage::SpacingType & inputSpacing = inputPtr->GetSpacing();
  const typename TInputImage::PointType &   inputOrigin = inputPtr->GetOrigin();
  const typename TInputImage::SizeType &    inputSize = inputPtr->GetLargestPossibleRegion().GetSize();
  const typename TInputImage::IndexType &   inputStartIndex = inputPtr->GetLargestPossibleRegion().GetIndex();

  typename TOutputImage::SpacingType outputSpacing(inputSpacing);
  typename TOutputImage::SizeType    outputSize(inputSize);
  typename TOutputImage::PointType   outputOrigin(inputOrigin);
  typename TOutputImage::IndexType   outputStartIndex;

  outputSize[1] = 2 * inputSize[1] + 1;

  if (this->m_Geometry->GetRadiusCylindricalDetector() == 0.)
  { // Flat panel detector


    // Compute sampling in psi
    const double D = m_Geometry->GetHelixSourceToDetectorDist();
    const double R = m_Geometry->GetHelixRadius();
    const double alpha_m = atan(0.5 * inputSpacing[0] * inputSize[0] / D);
    const double r = R * sin(alpha_m);
    const double psi_spacing = (M_PI + 2 * alpha_m) / (outputSize[1] - 1);

    outputSpacing[1] = psi_spacing;
    outputOrigin[1] = -0.5 * psi_spacing * (outputSize[1] - 1);

    typename TOutputImage::RegionType outputLargestPossibleRegion;
    outputLargestPossibleRegion.SetIndex(inputStartIndex);
    outputLargestPossibleRegion.SetSize(outputSize);
    outputPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);
    outputPtr->SetOrigin(outputOrigin);
    outputPtr->SetSpacing(outputSpacing);

  }
  else
  { // Cylindrical detector

    double radius = this->m_Geometry->GetRadiusCylindricalDetector();
    // Compute sampling in psi
    double alpha_m = 0.5 * inputSize[0] * inputSpacing[0] / radius;
    double psi_spacing = (M_PI + 2 * alpha_m) / (outputSize[1] - 1);
    outputSpacing[1] = psi_spacing;
    outputOrigin[1] = -0.5 * psi_spacing * (outputSize[1] - 1);

    typename TOutputImage::RegionType outputLargestPossibleRegion;
    outputLargestPossibleRegion.SetIndex(inputStartIndex);
    outputLargestPossibleRegion.SetSize(outputSize);
    outputPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);
    outputPtr->SetSpacing(outputSpacing);
    outputPtr->SetOrigin(outputOrigin);
  }
}

template <class TInputImage, class TOutputImage>
void
KatsevichForwardBinningImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() ITKv5_CONST
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull() || !this->m_Geometry->GetTheGeometryIsVerified())
    itkExceptionMacro(<< "Geometry has not been set or not been checked");
}


template <class TInputImage, class TOutputImage>
void
KatsevichForwardBinningImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread)
{

  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nProj = this->GetInput()->GetLargestPossibleRegion().GetSize(Dimension - 1);

  // Create interpolator, could be any interpolation
  using InterpolatorType = itk::LinearInterpolateImageFunction<TInputImage, float>;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetInputImage(this->GetInput());

  using OutputRegionIterator = itk::ImageRegionIteratorWithIndex<TOutputImage>;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Continuous index at which we interpolate
  itk::ContinuousIndex<double, Dimension - 1> pointProj;

  const typename TOutputImage::SpacingType & outputSpacing = this->GetOutput()->GetSpacing();
  const typename TOutputImage::PointType &   outputOrigin = this->GetOutput()->GetOrigin();
  const typename TOutputImage::SizeType &    outputSize = this->GetOutput()->GetLargestPossibleRegion().GetSize();
  const typename TOutputImage::IndexType & outputStartIndex = this->GetOutput()->GetLargestPossibleRegion().GetIndex();

  const typename TInputImage::SpacingType & inputSpacing = this->GetInput()->GetSpacing();
  const typename TInputImage::PointType &   inputOrigin = this->GetInput()->GetOrigin();
  const typename TInputImage::SizeType &    inputSize = this->GetInput()->GetLargestPossibleRegion().GetSize();


  // Parameters to compute psi and v
  const double R = m_Geometry->GetHelixRadius();
  const double D = m_Geometry->GetHelixSourceToDetectorDist();
  const double alpha_m = atan(0.5 * inputSpacing[0] * inputSize[0] / D);
  const double r = R * sin(alpha_m);

  const int    L = this->GetOutput()->GetLargestPossibleRegion().GetSize(1);
  const double delta_psi = (M_PI + 2 * alpha_m) / (L - 1);
  const double P = m_Geometry->GetHelixPitch();


  for (itOut.GoToBegin(); !itOut.IsAtEnd(); ++itOut)
  {
    if (this->m_Geometry->GetRadiusCylindricalDetector() == 0.) // Flat panel
    {
      typename TOutputImage::IndexType index = itOut.GetIndex();
      double                           psi = outputOrigin[1] + index[1] * outputSpacing[1];
      double                           u = outputOrigin[0] + index[0] * outputSpacing[0];
      double                           v = D * P / (2 * M_PI * R);
      if (psi != 0.)
      {
        v *= (psi + (psi / tan(psi)) * u / D);
      }
      else
      {
        v *= u / D;
      }
      itk::ContinuousIndex<float, TInputImage::ImageDimension> point;
      point[0] = index[0];
      point[1] = (v - inputOrigin[1]) / inputSpacing[1];
      point[2] = index[2];

      // Interpolate
      if (interpolator->IsInsideBuffer(point))
      {
        itOut.Set(interpolator->EvaluateAtContinuousIndex(point));
      }
    }
    else // Curved detector
    {

      double radius = this->m_Geometry->GetRadiusCylindricalDetector();
      if (radius != D)
      {
        itkExceptionMacro(<< "Implementation currently only handles cylindrical detector centered on source (i.e. "
                             "radius = source to det distance");
      }
      typename TOutputImage::IndexType index = itOut.GetIndex();
      double                           alpha = (outputOrigin[0] + index[0] * outputSpacing[0]) / radius;
      double                           psi = outputOrigin[1] + index[1] * outputSpacing[1];
      double                           v = D * P / (2 * M_PI * R);
      if (psi != 0.)
      {
        v *= (psi * cos(alpha) + (psi / tan(psi)) * sin(alpha));
      }
      else
      {
        v *= sin(alpha);
      }
      itk::ContinuousIndex<float, TInputImage::ImageDimension> point;
      point[0] = index[0];
      point[1] = (v - inputOrigin[1]) / inputSpacing[1];
      point[2] = index[2];

      // Interpolate
      if (interpolator->IsInsideBuffer(point))
      {
        itOut.Set(interpolator->EvaluateAtContinuousIndex(point));
      }
    }
  }
}

} // end namespace rtk

#endif
