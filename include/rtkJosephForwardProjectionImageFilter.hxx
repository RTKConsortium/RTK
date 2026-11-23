/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkJosephForwardProjectionImageFilter_hxx
#define rtkJosephForwardProjectionImageFilter_hxx

#include "math.h"


#include "rtkHomogeneousMatrix.h"
#include "rtkBoxShape.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"

#include <itkImageRegionIteratorWithIndex.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
JosephForwardProjectionImageFilter<TInputImage, TOutputImage>::JosephForwardProjectionImageFilter()
{
  this->SetStepFunction([](OutputPixelType & rayValue, EmptyRayState &, const StepSamples & samples) {
    rayValue += samples.Interpolate();
  });
  m_ProjectedValueAccumulation = [](const InputPixelType &  input,
                                    OutputPixelType &       output,
                                    const OutputPixelType & rayValue,
                                    const VectorType &      stepInMM,
                                    const PointType &,
                                    const VectorType &,
                                    const PointType &,
                                    const PointType &) { output = input + rayValue * stepInMM.GetNorm(); };
  this->DynamicMultiThreadingOff();
}


template <class TInputImage, class TOutputImage>
void
JosephForwardProjectionImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();

  if (this->GetInferiorClipImage().IsNotNull())
  {
    TClipImagePointer inputInferiorClipImage = const_cast<TClipImageType *>(this->GetInferiorClipImage().GetPointer());
    if (!inputInferiorClipImage)
      return;
    inputInferiorClipImage->SetRequestedRegion(inputInferiorClipImage->GetRequestedRegion());
  }

  if (this->GetSuperiorClipImage().IsNotNull())
  {
    TClipImagePointer inputSuperiorClipImage = const_cast<TClipImageType *>(this->GetSuperiorClipImage().GetPointer());
    if (!inputSuperiorClipImage)
      return;
    inputSuperiorClipImage->SetRequestedRegion(inputSuperiorClipImage->GetRequestedRegion());
  }
}


template <class TInputImage, class TOutputImage>
void
JosephForwardProjectionImageFilter<TInputImage, TOutputImage>::VerifyInputInformation() const
{
  using ImageBaseType = const itk::ImageBase<TInputImage::ImageDimension>;

  ImageBaseType * inputPtr1 = nullptr;

  itk::InputDataObjectConstIterator it(this);
  for (; !it.IsAtEnd(); ++it)
  {
    // Check whether the output is an image of the appropriate
    // dimension (use ProcessObject's version of the GetInput()
    // method since it returns the input as a pointer to a
    // DataObject as opposed to the subclass version which
    // static_casts the input to an TInputImage).
    if (it.GetName() == "Primary")
    {
      inputPtr1 = dynamic_cast<ImageBaseType *>(it.GetInput());
    }
    if (inputPtr1)
    {
      break;
    }
  }

  for (; !it.IsAtEnd(); ++it)
  {
    if (it.GetName() == "InferiorClipImage" || it.GetName() == "SuperiorClipImage")
    {
      auto * inputPtrN = dynamic_cast<ImageBaseType *>(it.GetInput());
      // Physical space computation only matters if we're using two
      // images, and not an image and a constant.
      if (inputPtrN)
      {
        // check that the image occupy the same physical space, and that
        // each index is at the same physical location

        // tolerance for origin and spacing depends on the size of pixel
        // tolerance for directions a fraction of the unit cube.
        const double coordinateTol = std::abs(Self::GetGlobalDefaultCoordinateTolerance() *
                                              inputPtr1->GetSpacing()[0]); // use first dimension spacing

        if (!inputPtr1->GetOrigin().GetVnlVector().is_equal(inputPtrN->GetOrigin().GetVnlVector(), coordinateTol) ||
            !inputPtr1->GetSpacing().GetVnlVector().is_equal(inputPtrN->GetSpacing().GetVnlVector(), coordinateTol) ||
            !inputPtr1->GetDirection().GetVnlMatrix().as_ref().is_equal(
              inputPtrN->GetDirection().GetVnlMatrix().as_ref(), Self::GetGlobalDefaultDirectionTolerance()))
        {
          std::ostringstream originString, spacingString, directionString;
          if (!inputPtr1->GetOrigin().GetVnlVector().is_equal(inputPtrN->GetOrigin().GetVnlVector(), coordinateTol))
          {
            originString.setf(std::ios::scientific);
            originString.precision(7);
            originString << "InputImage Origin: " << inputPtr1->GetOrigin() << ", InputImage" << it.GetName()
                         << " Origin: " << inputPtrN->GetOrigin() << std::endl;
            originString << "\tTolerance: " << coordinateTol << std::endl;
          }
          if (!inputPtr1->GetSpacing().GetVnlVector().is_equal(inputPtrN->GetSpacing().GetVnlVector(), coordinateTol))
          {
            spacingString.setf(std::ios::scientific);
            spacingString.precision(7);
            spacingString << "InputImage Spacing: " << inputPtr1->GetSpacing() << ", InputImage" << it.GetName()
                          << " Spacing: " << inputPtrN->GetSpacing() << std::endl;
            spacingString << "\tTolerance: " << coordinateTol << std::endl;
          }
          if (!inputPtr1->GetDirection().GetVnlMatrix().as_ref().is_equal(
                inputPtrN->GetDirection().GetVnlMatrix().as_ref(), Self::GetGlobalDefaultDirectionTolerance()))
          {
            directionString.setf(std::ios::scientific);
            directionString.precision(7);
            directionString << "InputImage Direction: " << inputPtr1->GetDirection() << ", InputImage" << it.GetName()
                            << " Direction: " << inputPtrN->GetDirection() << std::endl;
            directionString << "\tTolerance: " << Self::GetGlobalDefaultDirectionTolerance() << std::endl;
          }
          itkExceptionMacro(<< "Inputs do not occupy the same physical space! " << std::endl
                            << originString.str() << spacingString.str() << directionString.str());
        }
      }
    }
  }
}


template <class TInputImage, class TOutputImage>
void
JosephForwardProjectionImageFilter<TInputImage, TOutputImage>::ThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread,
  ThreadIdType                  itkNotUsed(threadId))
{
  m_RayCast(*this, outputRegionForThread);
}

template <class TInputImage, class TOutputImage>
template <class TStep, class TRayState>
void
JosephForwardProjectionImageFilter<TInputImage, TOutputImage>::RayCast(
  const OutputImageRegionType & outputRegionForThread,
  const TStep &                 step,
  const TRayState &             initialState)
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  ptrdiff_t          offsets[3];
  offsets[0] = 1;
  offsets[1] = this->GetInput(1)->GetBufferedRegion().GetSize()[0];
  offsets[2] =
    this->GetInput(1)->GetBufferedRegion().GetSize()[0] * this->GetInput(1)->GetBufferedRegion().GetSize()[1];
  const typename Superclass::GeometryType::ConstPointer geometry = this->GetGeometry();

  // Offsets are relative to the start of the buffer, originOffset is the offset of index (0,0,0)
  const InputPixelType * buffer = this->GetInput(1)->GetBufferPointer();
  const auto &           bufferIndex = this->GetInput(1)->GetBufferedRegion().GetIndex();
  const ptrdiff_t        originOffset =
    -offsets[0] * bufferIndex[0] - offsets[1] * bufferIndex[1] - offsets[2] * bufferIndex[2];

  // volPPToIndex maps the physical 3D coordinates of a point (in mm) to the
  // corresponding 3D volume index
  typename Superclass::GeometryType::ThreeDHomogeneousMatrixType volPPToIndex;
  volPPToIndex = GetPhysicalPointToIndexMatrix(this->GetInput(1));

  // Iterators on input and output projections
  using InputRegionIterator = ProjectionsRegionConstIteratorRayBased<TInputImage>;
  InputRegionIterator * itIn = nullptr;
  itIn = InputRegionIterator::New(this->GetInput(), outputRegionForThread, geometry, volPPToIndex);
  itk::ImageRegionIteratorWithIndex<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);
  using ClipImageIterator = itk::ImageRegionConstIterator<TClipImageType>;
  ClipImageIterator * itInferiorImage = nullptr;
  ClipImageIterator * itSuperiorImage = nullptr;
  if (this->GetInferiorClipImage().IsNotNull())
  {
    itInferiorImage = new ClipImageIterator(this->GetInferiorClipImage(), outputRegionForThread);
  }
  if (this->GetSuperiorClipImage().IsNotNull())
  {
    itSuperiorImage = new ClipImageIterator(this->GetSuperiorClipImage(), outputRegionForThread);
  }

  // Create intersection functions, one for each possible main direction
  auto                         box = BoxShape::New();
  typename BoxShape::PointType boxMin, boxMax;
  for (unsigned int i = 0; i < Dimension; i++)
  {
    boxMin[i] = this->GetInput(1)->GetBufferedRegion().GetIndex()[i];
    boxMax[i] =
      this->GetInput(1)->GetBufferedRegion().GetIndex()[i] + this->GetInput(1)->GetBufferedRegion().GetSize()[i] - 1;
    boxMax[i] *= 1. - itk::NumericTraits<BoxShape::ScalarType>::epsilon();
  }
  box->SetBoxMin(boxMin);
  box->SetBoxMax(boxMax);

  // m_InferiorClip and m_SuperiorClip are understood in the sense of a
  // source-to-pixel vector. Since we go from pixel-to-source, we invert them.
  double inferiorClip = 1. - m_SuperiorClip;
  double superiorClip = 1. - m_InferiorClip;

  // Go over each pixel of the projection
  for (size_t pix = 0; pix < outputRegionForThread.GetNumberOfPixels(); pix++, itIn->Next(), ++itOut)
  {
    typename InputRegionIterator::PointType  pixelPosition = itIn->GetPixelPosition();
    typename InputRegionIterator::VectorType dirVox = -itIn->GetSourceToPixel();

    // Select main direction
    unsigned int                  mainDir = 0;
    typename BoxShape::VectorType dirVoxAbs;
    for (unsigned int i = 0; i < Dimension; i++)
    {
      dirVoxAbs[i] = std::abs(dirVox[i]);
      if (dirVoxAbs[i] > dirVoxAbs[mainDir])
        mainDir = i;
    }

    if (this->GetInferiorClipImage().IsNotNull())
    {
      double superiorClipImage = 1 - itInferiorImage->Get();
      superiorClip = std::min(1. - m_InferiorClip, superiorClipImage);
      ++(*itInferiorImage);
    }
    if (this->GetSuperiorClipImage().IsNotNull())
    {
      double inferiorClipImage = 1 - itSuperiorImage->Get();
      inferiorClip = std::max(1. - m_SuperiorClip, inferiorClipImage);
      ++(*itSuperiorImage);
    }

    // Test if there is an intersection
    BoxShape::ScalarType infDist = NAN, supDist = NAN;
    bool                 isIntersectByRay = box->IsIntersectedByRay(pixelPosition, dirVox, infDist, supDist);
    // Clip the casting between source and pixel of the detector
    infDist = std::max(infDist, inferiorClip);
    supDist = std::min(supDist, superiorClip);

    if (isIntersectByRay && supDist > 0. && // check if detector after the source
        infDist <= 1. &&                    // check if detector after or in the volume
        supDist > infDist)
    {
      // Compute and sort intersections: (n)earest and (f)arthest (p)points
      typename BoxShape::PointType np = pixelPosition + infDist * dirVox;
      typename BoxShape::PointType fp = pixelPosition + supDist * dirVox;

      // Compute main nearest and farthest slice indices
      const int ns = itk::Math::rnd(np[mainDir]);
      const int fs = itk::Math::rnd(fp[mainDir]);

      // Determine the other two directions
      unsigned int notMainDirInf = (mainDir + 1) % Dimension;
      unsigned int notMainDirSup = (mainDir + 2) % Dimension;
      if (notMainDirInf > notMainDirSup)
        std::swap(notMainDirInf, notMainDirSup);

      const CoordinateType minx = box->GetBoxMin()[notMainDirInf];
      const CoordinateType miny = box->GetBoxMin()[notMainDirSup];
      const CoordinateType maxx = box->GetBoxMax()[notMainDirInf];
      const CoordinateType maxy = box->GetBoxMax()[notMainDirSup];

      // Init data offset to first pixel of slice ns
      const ptrdiff_t offsetx = offsets[notMainDirInf];
      const ptrdiff_t offsety = offsets[notMainDirSup];
      ptrdiff_t       offsetz = offsets[mainDir];
      ptrdiff_t       slice = originOffset + ns * offsetz;

      // Compute step size and go to first voxel
      CoordinateType       residualB = ns - np[mainDir];
      CoordinateType       residualE = fp[mainDir] - fs;
      const CoordinateType norm = itk::NumericTraits<CoordinateType>::One / dirVox[mainDir];
      CoordinateType       stepx = dirVox[notMainDirInf] * norm;
      CoordinateType       stepy = dirVox[notMainDirSup] * norm;
      if (np[mainDir] > fp[mainDir])
      {
        residualB *= -1;
        residualE *= -1;
        offsetz *= -1;
        stepx *= -1;
        stepy *= -1;
      }
      CoordinateType currentx = np[notMainDirInf] + residualB * stepx;
      CoordinateType currenty = np[notMainDirSup] + residualB * stepy;

      // Compute voxel to millimeters conversion
      typename BoxShape::VectorType stepMM;
      stepMM[notMainDirInf] = this->GetInput(1)->GetSpacing()[notMainDirInf] * stepx;
      stepMM[notMainDirSup] = this->GetInput(1)->GetSpacing()[notMainDirSup] * stepy;
      stepMM[mainDir] = this->GetInput(1)->GetSpacing()[mainDir];

      // Initialize the accumulation
      OutputPixelType rayValue = itk::NumericTraits<OutputPixelType>::ZeroValue();
      TRayState       state = initialState;
      const double    stepSizeInMM = stepMM.GetNorm();
      if (fs == ns) // If the voxel is a corner, we can skip most steps
      {
        step(rayValue,
             state,
             MakeStepSamples<true>(buffer,
                                   slice,
                                   currentx,
                                   currenty,
                                   offsetx,
                                   offsety,
                                   std::abs(fp[mainDir] - np[mainDir]),
                                   stepSizeInMM,
                                   minx,
                                   miny,
                                   maxx,
                                   maxy));
      }
      else
      {
        // First step
        step(rayValue,
             state,
             MakeStepSamples<true>(buffer,
                                   slice,
                                   currentx,
                                   currenty,
                                   offsetx,
                                   offsety,
                                   residualB + 0.5,
                                   stepSizeInMM,
                                   minx,
                                   miny,
                                   maxx,
                                   maxy));

        // Move to next main direction slice
        slice += offsetz;
        currentx += stepx;
        currenty += stepy;

        // Middle steps
        for (int i{ 0 }; i < std::abs(fs - ns) - 1; ++i)
        {
          step(rayValue,
               state,
               MakeStepSamples<false>(buffer, slice, currentx, currenty, offsetx, offsety, 1., stepSizeInMM));

          // Move to next main direction slice
          slice += offsetz;
          currentx += stepx;
          currenty += stepy;
        }

        // Last step
        step(rayValue,
             state,
             MakeStepSamples<true>(buffer,
                                   slice,
                                   currentx,
                                   currenty,
                                   offsetx,
                                   offsety,
                                   residualE + 0.5,
                                   stepSizeInMM,
                                   minx,
                                   miny,
                                   maxx,
                                   maxy));
      }
      // Accumulate
      m_ProjectedValueAccumulation(itIn->Get(), itOut.Value(), rayValue, stepMM, pixelPosition, dirVox, np, fp);
    }
    else
    {
      // This ray does not intersect the input image so just accumulate the
      // current pixel with the input value. Step and intersection points are
      // set to 0.
      m_ProjectedValueAccumulation(itIn->Get(), itOut.Value(), {}, {}, pixelPosition, dirVox, {}, {});
    }
  }
  delete itIn;
}

} // end namespace rtk

#endif
