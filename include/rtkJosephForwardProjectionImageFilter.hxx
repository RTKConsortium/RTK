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

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TProjectedValueAccumulation,
          class TSumAlongRay>
JosephForwardProjectionImageFilter<TInputImage,
                                   TOutputImage,
                                   TInterpolationWeightMultiplication,
                                   TProjectedValueAccumulation,
                                   TSumAlongRay>::JosephForwardProjectionImageFilter()
{
  this->DynamicMultiThreadingOff();
}


template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TProjectedValueAccumulation,
          class TSumAlongRay>
void
JosephForwardProjectionImageFilter<TInputImage,
                                   TOutputImage,
                                   TInterpolationWeightMultiplication,
                                   TProjectedValueAccumulation,
                                   TSumAlongRay>::GenerateInputRequestedRegion()
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


template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TProjectedValueAccumulation,
          class TSumAlongRay>
void
JosephForwardProjectionImageFilter<TInputImage,
                                   TOutputImage,
                                   TInterpolationWeightMultiplication,
                                   TProjectedValueAccumulation,
                                   TSumAlongRay>::VerifyInputInformation() const
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
        const double coordinateTol = itk::Math::abs(Self::GetGlobalDefaultCoordinateTolerance() *
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


template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TProjectedValueAccumulation,
          class TSumAlongRay>
void
JosephForwardProjectionImageFilter<TInputImage,
                                   TOutputImage,
                                   TInterpolationWeightMultiplication,
                                   TProjectedValueAccumulation,
                                   TSumAlongRay>::ThreadedGenerateData(const OutputImageRegionType &
                                                                                    outputRegionForThread,
                                                                       ThreadIdType threadId)
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  int                offsets[3];
  offsets[0] = 1;
  offsets[1] = this->GetInput(1)->GetBufferedRegion().GetSize()[0];
  offsets[2] =
    this->GetInput(1)->GetBufferedRegion().GetSize()[0] * this->GetInput(1)->GetBufferedRegion().GetSize()[1];
  const typename Superclass::GeometryType::ConstPointer geometry = this->GetGeometry();

  // beginBuffer is pointing at point with index (0,0,0) in memory, even if
  // it is not in the allocated memory
  const typename TInputImage::PixelType * beginBuffer =
    this->GetInput(1)->GetBufferPointer() - offsets[0] * this->GetInput(1)->GetBufferedRegion().GetIndex()[0] -
    offsets[1] * this->GetInput(1)->GetBufferedRegion().GetIndex()[1] -
    offsets[2] * this->GetInput(1)->GetBufferedRegion().GetIndex()[2];

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
  auto                          box = BoxShape::New();
  typename BoxShape::VectorType boxMin, boxMax;
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
  typename BoxShape::VectorType stepMM, np, fp;
  for (unsigned int pix = 0; pix < outputRegionForThread.GetNumberOfPixels(); pix++, itIn->Next(), ++itOut)
  {
    typename InputRegionIterator::PointType pixelPosition = itIn->GetPixelPosition();
    typename InputRegionIterator::PointType dirVox = -itIn->GetSourceToPixel();

    // Select main direction
    unsigned int                  mainDir = 0;
    typename BoxShape::VectorType dirVoxAbs;
    for (unsigned int i = 0; i < Dimension; i++)
    {
      dirVoxAbs[i] = itk::Math::abs(dirVox[i]);
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
    BoxShape::ScalarType nearDist = NAN, farDist = NAN;
    bool                 isIntersectByRay = box->IsIntersectedByRay(pixelPosition, dirVox, nearDist, farDist);
    // Clip the casting between source and pixel of the detector
    nearDist = std::max(nearDist, inferiorClip);
    farDist = std::min(farDist, superiorClip);

    if (isIntersectByRay && farDist > 0. && // check if detector after the source
        nearDist <= 1. &&                   // check if detector after or in the volume
        farDist > nearDist)
    {
      // Compute and sort intersections: (n)earest and (f)arthest (p)points
      np = pixelPosition + nearDist * dirVox;
      fp = pixelPosition + farDist * dirVox;

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

      // Init data pointers to first pixel of slice ns (i)nferior and (s)uperior (x|y) corner
      const int offsetx = offsets[notMainDirInf];
      const int offsety = offsets[notMainDirSup];
      int       offsetz = offsets[mainDir];

      const typename TInputImage::PixelType * pxiyi = beginBuffer + ns * offsetz;
      const typename TInputImage::PixelType * pxsyi = pxiyi + offsetx;
      const typename TInputImage::PixelType * pxiys = pxiyi + offsety;
      const typename TInputImage::PixelType * pxsys = pxsyi + offsety;

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
      stepMM[notMainDirInf] = this->GetInput(1)->GetSpacing()[notMainDirInf] * stepx;
      stepMM[notMainDirSup] = this->GetInput(1)->GetSpacing()[notMainDirSup] * stepy;
      stepMM[mainDir] = this->GetInput(1)->GetSpacing()[mainDir];

      // Initialize the accumulation
      typename TOutputImage::PixelType sum = itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue();

      typename TInputImage::PixelType volumeValue = itk::NumericTraits<typename TInputImage::PixelType>::ZeroValue();
      if (fs == ns) // If the voxel is a corner, we can skip most steps
      {
        volumeValue = BilinearInterpolationOnBorders(threadId,
                                                     itk::Math::abs(fp[mainDir] - np[mainDir]),
                                                     pxiyi,
                                                     pxsyi,
                                                     pxiys,
                                                     pxsys,
                                                     currentx,
                                                     currenty,
                                                     offsetx,
                                                     offsety,
                                                     minx,
                                                     miny,
                                                     maxx,
                                                     maxy);
        m_SumAlongRay(threadId, sum, volumeValue, stepMM);
      }
      else
      {
        // First step
        volumeValue = BilinearInterpolationOnBorders(threadId,
                                                     residualB + 0.5,
                                                     pxiyi,
                                                     pxsyi,
                                                     pxiys,
                                                     pxsys,
                                                     currentx,
                                                     currenty,
                                                     offsetx,
                                                     offsety,
                                                     minx,
                                                     miny,
                                                     maxx,
                                                     maxy);
        m_SumAlongRay(threadId, sum, volumeValue, stepMM);

        // Move to next main direction slice
        pxiyi += offsetz;
        pxsyi += offsetz;
        pxiys += offsetz;
        pxsys += offsetz;
        currentx += stepx;
        currenty += stepy;

        // Middle steps
        for (int i{ 0 }; i < itk::Math::abs(fs - ns) - 1; ++i)
        {
          volumeValue =
            BilinearInterpolation(threadId, 1., pxiyi, pxsyi, pxiys, pxsys, currentx, currenty, offsetx, offsety);
          m_SumAlongRay(threadId, sum, volumeValue, stepMM);

          // Move to next main direction slice
          pxiyi += offsetz;
          pxsyi += offsetz;
          pxiys += offsetz;
          pxsys += offsetz;
          currentx += stepx;
          currenty += stepy;
        }

        // Last step
        volumeValue = BilinearInterpolationOnBorders(threadId,
                                                     residualE + 0.5,
                                                     pxiyi,
                                                     pxsyi,
                                                     pxiys,
                                                     pxsys,
                                                     currentx,
                                                     currenty,
                                                     offsetx,
                                                     offsety,
                                                     minx,
                                                     miny,
                                                     maxx,
                                                     maxy);
        m_SumAlongRay(threadId, sum, volumeValue, stepMM);
      }
      // Accumulate
      m_ProjectedValueAccumulation(threadId, itIn->Get(), itOut.Value(), sum, stepMM, pixelPosition, dirVox, np, fp);
    }
    else
      m_ProjectedValueAccumulation(
        threadId, itIn->Get(), itOut.Value(), {}, pixelPosition, pixelPosition, dirVox, pixelPosition, pixelPosition);
  }
  delete itIn;
}

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TProjectedValueAccumulation,
          class TSumAlongRay>
typename JosephForwardProjectionImageFilter<TInputImage,
                                            TOutputImage,
                                            TInterpolationWeightMultiplication,
                                            TProjectedValueAccumulation,
                                            TSumAlongRay>::OutputPixelType
JosephForwardProjectionImageFilter<TInputImage,
                                   TOutputImage,
                                   TInterpolationWeightMultiplication,
                                   TProjectedValueAccumulation,
                                   TSumAlongRay>::BilinearInterpolation(const ThreadIdType     threadId,
                                                                        const double           stepLengthInVoxel,
                                                                        const InputPixelType * pxiyi,
                                                                        const InputPixelType * pxsyi,
                                                                        const InputPixelType * pxiys,
                                                                        const InputPixelType * pxsys,
                                                                        const CoordinateType   x,
                                                                        const CoordinateType   y,
                                                                        const int              ox,
                                                                        const int              oy)
{
  int            ix = itk::Math::floor(x);
  int            iy = itk::Math::floor(y);
  int            idx = ix * ox + iy * oy;
  CoordinateType lx = x - ix;
  CoordinateType ly = y - iy;
  CoordinateType lxc = 1. - lx;
  CoordinateType lyc = 1. - ly;
  return (m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * lyc, pxiyi, idx) +
          m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lx * lyc, pxsyi, idx) +
          m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * ly, pxiys, idx) +
          m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lx * ly, pxsys, idx));
  /* Alternative slower solution
  const unsigned int   ix = itk::Math::Floor(x);
  const unsigned int   iy = itk::Math::Floor(y);
  const unsigned int   idx = ix*ox + iy*oy;
  const CoordinateType a = p1[idx];
  const CoordinateType b = p2[idx] - a;
  const CoordinateType c = p3[idx] - a;
  const CoordinateType lx = x-ix;
  const CoordinateType ly = y-iy;
  const CoordinateType d = p4[idx] - a - b - c;
  return a + b*lx + c*ly + d*lx*ly;
*/
}

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TProjectedValueAccumulation,
          class TSumAlongRay>
typename JosephForwardProjectionImageFilter<TInputImage,
                                            TOutputImage,
                                            TInterpolationWeightMultiplication,
                                            TProjectedValueAccumulation,
                                            TSumAlongRay>::OutputPixelType
JosephForwardProjectionImageFilter<TInputImage,
                                   TOutputImage,
                                   TInterpolationWeightMultiplication,
                                   TProjectedValueAccumulation,
                                   TSumAlongRay>::BilinearInterpolationOnBorders(const ThreadIdType threadId,
                                                                                 const double       stepLengthInVoxel,
                                                                                 const InputPixelType * pxiyi,
                                                                                 const InputPixelType * pxsyi,
                                                                                 const InputPixelType * pxiys,
                                                                                 const InputPixelType * pxsys,
                                                                                 const CoordinateType   x,
                                                                                 const CoordinateType   y,
                                                                                 const int              ox,
                                                                                 const int              oy,
                                                                                 const CoordinateType   minx,
                                                                                 const CoordinateType   miny,
                                                                                 const CoordinateType   maxx,
                                                                                 const CoordinateType   maxy)
{
  int            ix = itk::Math::floor(x);
  int            iy = itk::Math::floor(y);
  int            idx = ix * ox + iy * oy;
  CoordinateType lx = x - ix;
  CoordinateType ly = y - iy;
  CoordinateType lxc = 1. - lx;
  CoordinateType lyc = 1. - ly;

  int offset_xi = 0;
  int offset_yi = 0;
  int offset_xs = 0;
  int offset_ys = 0;

  OutputPixelType result = itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue();
  if (ix < minx)
    offset_xi = ox;
  if (iy < miny)
    offset_yi = oy;
  if (ix >= maxx)
    offset_xs = -ox;
  if (iy >= maxy)
    offset_ys = -oy;

  result +=
    m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * lyc, pxiyi, idx + offset_xi + offset_yi);
  result +=
    m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * ly, pxiys, idx + offset_xi + offset_ys);
  result +=
    m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lx * lyc, pxsyi, idx + offset_xs + offset_yi);
  result +=
    m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lx * ly, pxsys, idx + offset_xs + offset_ys);

  result *= stepLengthInVoxel;

  return (result);
}
} // end namespace rtk

#endif
