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

#ifndef rtkKatsevichBackwardBinningImageFilter_hxx
#define rtkKatsevichBackwardBinningImageFilter_hxx

#include "math.h"

#include <rtkKatsevichBackwardBinningImageFilter.h>

#include <rtkHomogeneousMatrix.h>

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkPixelTraits.h>

namespace rtk
{

template <typename TInputImage, typename TOutputImage>
void
KatsevichBackwardBinningImageFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
{
  // Update output size and origin.
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
  typename TOutputImage::IndexType          outputStartIndex;

  typename TOutputImage::SpacingType outputSpacing(inputSpacing);
  typename TOutputImage::SizeType    outputSize(inputSize);
  typename TOutputImage::PointType   outputOrigin(inputOrigin);

  outputSize[1] = (int)(inputSize[1] - 1) / 2;
  outputSpacing[1] = outputSpacing[0];
  outputOrigin[1] = -0.5 * outputSpacing[1] * (outputSize[1] - 1);

  typename TOutputImage::RegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion.SetIndex(inputStartIndex);
  outputLargestPossibleRegion.SetSize(outputSize);
  outputPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);
  outputPtr->SetOrigin(outputOrigin);
  outputPtr->SetSpacing(outputSpacing);
}

template <class TInputImage, class TOutputImage>
void
KatsevichBackwardBinningImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() ITKv5_CONST
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull() || !this->m_Geometry->GetTheGeometryIsVerified())
    itkExceptionMacro(<< "Geometry has not been set or not been verified");
}


template <class TInputImage, class TOutputImage>
void
KatsevichBackwardBinningImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread)
{
  const unsigned int Dimension = TInputImage::ImageDimension;

  // Iterator over output projections (in v)
  OutputSliceIteratorType itOut(this->GetOutput(), outputRegionForThread);
  itOut.SetFirstDirection(1);
  itOut.SetSecondDirection(0);

  // Detector details
  unsigned int nbproj = outputRegionForThread.GetSize()[2];
  unsigned int det_nu = outputRegionForThread.GetSize()[0];
  unsigned int det_nv = outputRegionForThread.GetSize()[1];

  unsigned int idx_proj = outputRegionForThread.GetIndex()[2];

  InputImageRegionType requiredInputRegion;
  InputImageIndexType  inIndex = this->GetInput()->GetLargestPossibleRegion().GetIndex();
  InputImageSizeType   inSize = this->GetInput()->GetLargestPossibleRegion().GetSize();
  inIndex[2] = idx_proj;
  inSize[2] = nbproj;
  requiredInputRegion.SetIndex(inIndex);
  requiredInputRegion.SetSize(inSize);

  InputSliceIteratorType itIn(this->GetInput(), requiredInputRegion);
  itIn.SetFirstDirection(1);
  itIn.SetSecondDirection(0);

  double u = 0;
  double v_current = 0;
  double v_psi = 0;
  double psi = 0;

  const double u_spacing = this->GetInput()->GetSpacing()[0];
  const double psi_spacing = this->GetInput()->GetSpacing()[1];
  const double v_spacing = this->GetOutput()->GetSpacing()[1];
  const double u_orig = this->GetInput()->GetOrigin()[0];
  const double psi_orig = this->GetInput()->GetOrigin()[1];
  const double v_orig = this->GetOutput()->GetOrigin()[1];

  // Geometry details
  const double D = m_Geometry->GetHelixSourceToDetectorDist();
  const double P = m_Geometry->GetHelixPitch();
  const double R = m_Geometry->GetHelixRadius();

  const double alpha_m = atan(0.5 * u_spacing * det_nu / D); // Consistent with Forward binning.
  const double r = R * sin(alpha_m);
  const int    L = this->GetInput()->GetLargestPossibleRegion().GetSize(1);

  unsigned int idx_v = 0;
  double       coeff_s = 0., coeff_i = 0.;
  itIn.GoToBegin();
  itOut.GoToBegin();
  while (!itIn.IsAtEnd())
  {
    while (!itIn.IsAtEndOfSlice())
    {
      // Vectors of output values and coeffs accumulator
      std::vector<float> coeffs(det_nv, 0.);
      std::vector<float> outputVals(det_nv, 0.);
      while (!itIn.IsAtEndOfLine())
      {
        const typename TInputImage::IndexType idx = itIn.GetIndex();
        psi = psi_orig + idx[1] * psi_spacing;
        u = u_orig + idx[0] * u_spacing;

        // Compute v corresponding to current psi value
        if (psi != 0)
          v_psi = (psi + (psi / tan(psi)) * (u / D)) * D * P / (R * 2 * M_PI);
        else
          v_psi = (u / D) * D * P / (R * 2 * M_PI);

        // Raise error if v outside detector range
        if (v_psi < v_orig - 0.5 * v_spacing || v_psi >= v_orig + v_spacing * (det_nv - 1 + 0.5))
        {
          itkExceptionMacro(<< "The v_k(psi) value is outside v-range. v_psi :" << v_psi << " v_orig, v_spacing "
                            << v_orig << " " << v_spacing << "psi : " << psi << " det_nv : " << det_nv);
        }
        else if (v_psi < v_orig) // v is before first pixel center but within the detector range
        {
          idx_v = itk::Math::floor((v_psi - v_orig) / v_spacing);
          v_current = v_orig + idx_v * v_spacing;
          coeff_s = (v_psi - v_current) / v_spacing;
          // Accumulate
          outputVals[0] += coeff_s * itIn.Get();
          coeffs[0] += coeff_s;
        }
        else if (v_psi >=
                 v_orig + v_spacing * (det_nv - 1)) // v is after last pixel center but within the detector range
        {
          idx_v = itk::Math::floor((v_psi - v_orig) / v_spacing);
          v_current = v_orig + idx_v * v_spacing;
          coeff_s = (v_psi - v_current) / v_spacing;
          coeff_i = 1 - coeff_s;
          outputVals[det_nv - 1] += coeff_i * itIn.Get();
          coeffs[det_nv - 1] += coeff_i;
        }
        else // standard case
        {
          idx_v = itk::Math::floor((v_psi - v_orig) / v_spacing);
          v_current = v_orig + idx_v * v_spacing;
          coeff_s = (v_psi - v_current) / v_spacing;
          coeff_i = 1 - coeff_s;

          // Accumulate
          outputVals[idx_v] += coeff_i * itIn.Get();
          outputVals[idx_v + 1] += coeff_s * itIn.Get();
          coeffs[idx_v] += coeff_i;
          coeffs[idx_v + 1] += coeff_s;
        }
        // Move iterator forward
        ++itIn;
      }

      // Divide output by the sum of all coeffs contributions.
      while (!itOut.IsAtEndOfLine())
      {
        OutputImageIndexType index = itOut.GetIndex();
        int                  idx = index[1];
        if (coeffs[idx] == 0.)
        {
          itOut.Set(0.);
        }
        else
        {
          itOut.Set(outputVals[idx] / coeffs[idx]);
        }
        if (this->m_Geometry->GetRadiusCylindricalDetector() != 0.)
        {
          double cosalpha = cos((u_orig + index[0] * u_spacing) / this->m_Geometry->GetRadiusCylindricalDetector());
          itOut.Set(itOut.Get() * cosalpha);
        }

        ++itOut;
      }
      itIn.NextLine();
      itOut.NextLine();
    }
    itIn.NextSlice();
    itOut.NextSlice();
  }
}

} // end namespace rtk

#endif
