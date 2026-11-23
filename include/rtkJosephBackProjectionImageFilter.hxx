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

#ifndef rtkJosephBackProjectionImageFilter_hxx
#define rtkJosephBackProjectionImageFilter_hxx

#include "math.h"


#include "rtkHomogeneousMatrix.h"
#include "rtkBoxShape.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
JosephBackProjectionImageFilter<TInputImage, TOutputImage>::JosephBackProjectionImageFilter()
{
  this->SetStepFunction([](OutputPixelType * volume, EmptyRayState &, const StepSamples & samples) {
    for (unsigned int k = 0; k < 4; ++k)
    {
      volume[samples.offsets[k]] += static_cast<OutputPixelType>(samples.projectionValue * samples.weights[k] *
                                                                 samples.stepSizeInMM * samples.stepLengthInVoxel);
    }
  });
}

template <class TInputImage, class TOutputImage>
void
JosephBackProjectionImageFilter<TInputImage, TOutputImage>::GenerateData()
{
  m_BackProject(*this);
}

template <class TInputImage, class TOutputImage>
template <class TStep, class TRayState>
void
JosephBackProjectionImageFilter<TInputImage, TOutputImage>::BackProject(const TStep &     step,
                                                                        const TRayState & initialState)
{
  // Allocate the output image
  this->AllocateOutputs();

  const unsigned int               Dimension = TInputImage::ImageDimension;
  typename TInputImage::RegionType buffReg = this->GetInput(1)->GetBufferedRegion();
  ptrdiff_t                        offsets[3];
  offsets[0] = 1;
  offsets[1] = this->GetInput(0)->GetBufferedRegion().GetSize()[0];
  offsets[2] =
    this->GetInput(0)->GetBufferedRegion().GetSize()[0] * this->GetInput(0)->GetBufferedRegion().GetSize()[1];

  const auto * geometry = dynamic_cast<const GeometryType *>(this->GetGeometry());
  if (!geometry)
  {
    itkGenericExceptionMacro(<< "Error, ThreeDCircularProjectionGeometry expected");
  }

  // Offsets are relative to the start of the buffer, originOffset is the offset of index (0,0,0)
  OutputPixelType * buffer = this->GetOutput()->GetBufferPointer();
  const auto &      bufferIndex = this->GetOutput()->GetBufferedRegion().GetIndex();
  const ptrdiff_t   originOffset =
    -offsets[0] * bufferIndex[0] - offsets[1] * bufferIndex[1] - offsets[2] * bufferIndex[2];

  // volPPToIndex maps the physical 3D coordinates of a point (in mm) to the
  // corresponding 3D volume index
  typename GeometryType::ThreeDHomogeneousMatrixType volPPToIndex;
  volPPToIndex = GetPhysicalPointToIndexMatrix(this->GetInput(0));

  // Initialize output region with input region in case the filter is not in
  // place
  if (this->GetInput() != this->GetOutput())
  {
    // Iterators on volume input and output
    using InputRegionIterator = itk::ImageRegionConstIterator<TInputImage>;
    InputRegionIterator itVolIn(this->GetInput(0), this->GetInput()->GetBufferedRegion());

    itk::ImageRegionIteratorWithIndex<TOutputImage> itVolOut(this->GetOutput(), this->GetInput()->GetBufferedRegion());

    while (!itVolIn.IsAtEnd())
    {
      itVolOut.Set(itVolIn.Get());
      ++itVolIn;
      ++itVolOut;
    }
  }

  // Iterators on projections input
  using InputRegionIterator = ProjectionsRegionConstIteratorRayBased<TInputImage>;
  InputRegionIterator * itIn = nullptr;
  itIn = InputRegionIterator::New(this->GetInput(1), buffReg, geometry, volPPToIndex);

  // Create intersection functions, one for each possible main direction
  auto                         box = BoxShape::New();
  typename BoxShape::PointType boxMin, boxMax;
  for (unsigned int i = 0; i < Dimension; i++)
  {
    boxMin[i] = this->GetOutput()->GetRequestedRegion().GetIndex()[i];
    boxMax[i] =
      this->GetOutput()->GetRequestedRegion().GetIndex()[i] + this->GetOutput()->GetRequestedRegion().GetSize()[i] - 1;
    boxMax[i] *= 1. - itk::NumericTraits<BoxShape::ScalarType>::epsilon();
  }
  box->SetBoxMin(boxMin);
  box->SetBoxMax(boxMax);

  // m_InferiorClip and m_SuperiorClip are understood in the sense of a
  // source-to-pixel vector. Since we go from pixel-to-source, we invert them.
  double inferiorClip = 1. - m_SuperiorClip;
  double superiorClip = 1. - m_InferiorClip;

  // Go over each pixel of the projection
  typename BoxShape::VectorType stepMM;
  typename BoxShape::PointType  np, fp;
  for (size_t pix = 0; pix < buffReg.GetNumberOfPixels(); pix++, itIn->Next())
  {
    typename InputRegionIterator::PointType  pixelPosition = itIn->GetPixelPosition();
    typename InputRegionIterator::VectorType dirVox = -itIn->GetSourceToPixel();

    // Select main direction
    unsigned int         mainDir = 0;
    BoxShape::VectorType dirVoxAbs;
    for (unsigned int i = 0; i < Dimension; i++)
    {
      dirVoxAbs[i] = std::abs(dirVox[i]);
      if (dirVoxAbs[i] > dirVoxAbs[mainDir])
        mainDir = i;
    }

    // Test if there is an intersection
    BoxShape::ScalarType infDist = NAN, supDist = NAN;
    if (box->IsIntersectedByRay(pixelPosition, dirVox, infDist, supDist) &&
        supDist >= 0. && // check if detector after the source
        infDist <= 1.)   // check if detector after or in the volume
    {
      // Clip the casting between source and pixel of the detector
      infDist = std::max(infDist, inferiorClip);
      supDist = std::min(supDist, superiorClip);

      // Compute and sort intersections: (n)earest and (f)arthest (p)points
      np = pixelPosition + infDist * dirVox;
      fp = pixelPosition + supDist * dirVox;

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
      stepMM[notMainDirInf] = this->GetInput(0)->GetSpacing()[notMainDirInf] * stepx;
      stepMM[notMainDirSup] = this->GetInput(0)->GetSpacing()[notMainDirSup] * stepy;
      stepMM[mainDir] = this->GetInput(0)->GetSpacing()[mainDir];

      TRayState            state = initialState;
      const InputPixelType projectionValue = itIn->Value();
      const double         stepSizeInMM = stepMM.GetNorm();
      if (fs == ns) // If the voxel is a corner, we can skip most steps
      {
        step(buffer,
             state,
             MakeStepSamples<true>(projectionValue,
                                   pix,
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
        step(buffer,
             state,
             MakeStepSamples<true>(projectionValue,
                                   pix,
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
          step(buffer,
               state,
               MakeStepSamples<false>(
                 projectionValue, pix, slice, currentx, currenty, offsetx, offsety, 1., stepSizeInMM));

          // Move to next main direction slice
          slice += offsetz;
          currentx += stepx;
          currenty += stepy;
        }

        // Last step
        step(buffer,
             state,
             MakeStepSamples<true>(projectionValue,
                                   pix,
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
    }
  }
  delete itIn;
}

} // end namespace rtk

#endif
