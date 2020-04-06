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

#ifndef rtkJosephBackProjectionImageFilter_hxx
#define rtkJosephBackProjectionImageFilter_hxx

#include "math.h"

#include "rtkJosephBackProjectionImageFilter.h"

#include "rtkHomogeneousMatrix.h"
#include "rtkBoxShape.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>

namespace rtk
{

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TSplatWeightMultiplication,
          class TSumAlongRay>
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TInterpolationWeightMultiplication,
                                TSplatWeightMultiplication,
                                TSumAlongRay>::JosephBackProjectionImageFilter() = default;

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TSplatWeightMultiplication,
          class TSumAlongRay>
void
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TInterpolationWeightMultiplication,
                                TSplatWeightMultiplication,
                                TSumAlongRay>::GenerateData()
{
  // Allocate the output image
  this->AllocateOutputs();

  const unsigned int               Dimension = TInputImage::ImageDimension;
  typename TInputImage::RegionType buffReg = this->GetInput(1)->GetBufferedRegion();
  int                              offsets[3];
  offsets[0] = 1;
  offsets[1] = this->GetInput(0)->GetBufferedRegion().GetSize()[0];
  offsets[2] =
    this->GetInput(0)->GetBufferedRegion().GetSize()[0] * this->GetInput(0)->GetBufferedRegion().GetSize()[1];

  const auto * geometry = dynamic_cast<const GeometryType *>(this->GetGeometry());
  if (!geometry)
  {
    itkGenericExceptionMacro(<< "Error, ThreeDCircularProjectionGeometry expected");
  }

  // beginBuffer is pointing at point with index (0,0,0) in memory, even if
  // it is not in the allocated memory
  typename TOutputImage::PixelType * beginBuffer = this->GetOutput()->GetBufferPointer() -
                                                   offsets[0] * this->GetOutput()->GetBufferedRegion().GetIndex()[0] -
                                                   offsets[1] * this->GetOutput()->GetBufferedRegion().GetIndex()[1] -
                                                   offsets[2] * this->GetOutput()->GetBufferedRegion().GetIndex()[2];

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

    using OutputRegionIterator = itk::ImageRegionIteratorWithIndex<TOutputImage>;
    OutputRegionIterator itVolOut(this->GetOutput(), this->GetInput()->GetBufferedRegion());

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
  typename BoxShape::Pointer    box = BoxShape::New();
  typename BoxShape::VectorType boxMin, boxMax;
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
  typename BoxShape::VectorType stepMM, np, fp;
  for (unsigned int pix = 0; pix < buffReg.GetNumberOfPixels(); pix++, itIn->Next())
  {
    typename InputRegionIterator::PointType sourcePosition = itIn->GetSourcePosition();
    typename InputRegionIterator::PointType dirVox = itIn->GetSourceToPixel();

    // Select main direction
    unsigned int         mainDir = 0;
    BoxShape::VectorType dirVoxAbs;
    for (unsigned int i = 0; i < Dimension; i++)
    {
      dirVoxAbs[i] = itk::Math::abs(dirVox[i]);
      if (dirVoxAbs[i] > dirVoxAbs[mainDir])
        mainDir = i;
    }

    // Test if there is an intersection
    BoxShape::ScalarType nearDist = NAN, farDist = NAN;
    if (box->IsIntersectedByRay(sourcePosition, dirVox, nearDist, farDist) &&
        farDist >= 0. && // check if detector after the source
        nearDist <= 1.)  // check if detector after or in the volume
    {
      // Clip the casting between source and pixel of the detector
      nearDist = std::max(nearDist, inferiorClip);
      farDist = std::min(farDist, superiorClip);

      // Compute and sort intersections: (n)earest and (f)arthest (p)points
      np = sourcePosition + nearDist * dirVox;
      fp = sourcePosition + farDist * dirVox;
      if (np[mainDir] > fp[mainDir])
        std::swap(np, fp);

      // Compute main nearest and farthest slice indices
      const int ns = itk::Math::rnd(np[mainDir]);
      const int fs = itk::Math::rnd(fp[mainDir]);

      // Determine the other two directions
      unsigned int notMainDirInf = (mainDir + 1) % Dimension;
      unsigned int notMainDirSup = (mainDir + 2) % Dimension;
      if (notMainDirInf > notMainDirSup)
        std::swap(notMainDirInf, notMainDirSup);

      const CoordRepType minx = box->GetBoxMin()[notMainDirInf];
      const CoordRepType miny = box->GetBoxMin()[notMainDirSup];
      const CoordRepType maxx = box->GetBoxMax()[notMainDirInf];
      const CoordRepType maxy = box->GetBoxMax()[notMainDirSup];

      // Init data pointers to first pixel of slice ns (i)nferior and (s)uperior (x|y) corner
      const int        offsetx = offsets[notMainDirInf];
      const int        offsety = offsets[notMainDirSup];
      const int        offsetz = offsets[mainDir];
      OutputPixelType *pxiyi = nullptr, *pxsyi = nullptr, *pxiys = nullptr, *pxsys = nullptr;

      pxiyi = beginBuffer + ns * offsetz;
      pxsyi = pxiyi + offsetx;
      pxiys = pxiyi + offsety;
      pxsys = pxsyi + offsety;

      // Compute step size and go to first voxel
      const CoordRepType residual = ns - np[mainDir];
      const CoordRepType norm = 1 / dirVox[mainDir];
      const CoordRepType stepx = dirVox[notMainDirInf] * norm;
      const CoordRepType stepy = dirVox[notMainDirSup] * norm;
      CoordRepType       currentx = np[notMainDirInf] + residual * stepx;
      CoordRepType       currenty = np[notMainDirSup] + residual * stepy;

      // Compute voxel to millimeters conversion
      stepMM[notMainDirInf] = this->GetInput(0)->GetSpacing()[notMainDirInf] * stepx;
      stepMM[notMainDirSup] = this->GetInput(0)->GetSpacing()[notMainDirSup] * stepy;
      stepMM[mainDir] = this->GetInput(0)->GetSpacing()[mainDir];

      typename TOutputImage::PixelType attenuationRay =
        itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue();
      bool isNewRay = true;
      if (fs == ns) // If the voxel is a corner, we can skip most steps
      {
        attenuationRay += BilinearInterpolationOnBorders(fp[mainDir] - np[mainDir],
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
        const typename TInputImage::PixelType & rayValue =
          m_SumAlongRay(itIn->Value(), attenuationRay, stepMM, isNewRay);
        BilinearSplatOnBorders(rayValue,
                               fp[mainDir] - np[mainDir],
                               stepMM.GetNorm(),
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
      }
      else
      {
        // First step
        attenuationRay += BilinearInterpolationOnBorders(
          residual + 0.5, pxiyi, pxsyi, pxiys, pxsys, currentx, currenty, offsetx, offsety, minx, miny, maxx, maxy);

        const typename TInputImage::PixelType & rayValueF =
          m_SumAlongRay(itIn->Value(), attenuationRay, stepMM, isNewRay);

        BilinearSplatOnBorders(rayValueF,
                               residual + 0.5,
                               stepMM.GetNorm(),
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

        // Move to next main direction slice
        pxiyi += offsetz;
        pxsyi += offsetz;
        pxiys += offsetz;
        pxsys += offsetz;
        currentx += stepx;
        currenty += stepy;

        // Middle steps
        for (int i = ns + 1; i < fs; i++)
        {
          attenuationRay +=
            BilinearInterpolation(1.0, pxiyi, pxsyi, pxiys, pxsys, currentx, currenty, offsetx, offsety);

          const typename TInputImage::PixelType & rayValueM =
            m_SumAlongRay(itIn->Value(), attenuationRay, stepMM, isNewRay);

          BilinearSplat(
            rayValueM, 1.0, stepMM.GetNorm(), pxiyi, pxsyi, pxiys, pxsys, currentx, currenty, offsetx, offsety);

          // Move to next main direction slice
          pxiyi += offsetz;
          pxsyi += offsetz;
          pxiys += offsetz;
          pxsys += offsetz;
          currentx += stepx;
          currenty += stepy;
        }

        // Last step
        attenuationRay += BilinearInterpolationOnBorders(fp[mainDir] - fs + 0.5,
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

        const typename TInputImage::PixelType & rayValueE =
          m_SumAlongRay(itIn->Value(), attenuationRay, stepMM, isNewRay);

        BilinearSplatOnBorders(rayValueE,
                               fp[mainDir] - fs + 0.5,
                               stepMM.GetNorm(),
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
      }
    }
  }
  delete itIn;
}

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TSplatWeightMultiplication,
          class TSumAlongRay>
void
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TInterpolationWeightMultiplication,
                                TSplatWeightMultiplication,
                                TSumAlongRay>::BilinearSplat(const InputPixelType & rayValue,
                                                             const double           stepLengthInVoxel,
                                                             const double           voxelSize,
                                                             OutputPixelType *      pxiyi,
                                                             OutputPixelType *      pxsyi,
                                                             OutputPixelType *      pxiys,
                                                             OutputPixelType *      pxsys,
                                                             const double           x,
                                                             const double           y,
                                                             const int              ox,
                                                             const int              oy)
{
  int          ix = itk::Math::floor(x);
  int          iy = itk::Math::floor(y);
  int          idx = ix * ox + iy * oy;
  CoordRepType lx = x - ix;
  CoordRepType ly = y - iy;
  CoordRepType lxc = 1. - lx;
  CoordRepType lyc = 1. - ly;

  m_SplatWeightMultiplication(rayValue, pxiyi[idx], stepLengthInVoxel, voxelSize, lxc * lyc);
  m_SplatWeightMultiplication(rayValue, pxsyi[idx], stepLengthInVoxel, voxelSize, lx * lyc);
  m_SplatWeightMultiplication(rayValue, pxiys[idx], stepLengthInVoxel, voxelSize, lxc * ly);
  m_SplatWeightMultiplication(rayValue, pxsys[idx], stepLengthInVoxel, voxelSize, lx * ly);
}

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TSplatWeightMultiplication,
          class TSumAlongRay>
void
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TInterpolationWeightMultiplication,
                                TSplatWeightMultiplication,
                                TSumAlongRay>::BilinearSplatOnBorders(const InputPixelType & rayValue,
                                                                      const double           stepLengthInVoxel,
                                                                      const double           voxelSize,
                                                                      OutputPixelType *      pxiyi,
                                                                      OutputPixelType *      pxsyi,
                                                                      OutputPixelType *      pxiys,
                                                                      OutputPixelType *      pxsys,
                                                                      const double           x,
                                                                      const double           y,
                                                                      const int              ox,
                                                                      const int              oy,
                                                                      const CoordRepType     minx,
                                                                      const CoordRepType     miny,
                                                                      const CoordRepType     maxx,
                                                                      const CoordRepType     maxy)
{
  int          ix = itk::Math::floor(x);
  int          iy = itk::Math::floor(y);
  int          idx = ix * ox + iy * oy;
  CoordRepType lx = x - ix;
  CoordRepType ly = y - iy;
  CoordRepType lxc = 1. - lx;
  CoordRepType lyc = 1. - ly;

  int offset_xi = 0;
  int offset_yi = 0;
  int offset_xs = 0;
  int offset_ys = 0;

  if (ix < minx)
    offset_xi = ox;
  if (iy < miny)
    offset_yi = oy;
  if (ix >= maxx)
    offset_xs = -ox;
  if (iy >= maxy)
    offset_ys = -oy;

  m_SplatWeightMultiplication(rayValue, pxiyi[idx + offset_xi + offset_yi], stepLengthInVoxel, voxelSize, lxc * lyc);
  m_SplatWeightMultiplication(rayValue, pxiys[idx + offset_xi + offset_ys], stepLengthInVoxel, voxelSize, lxc * ly);
  m_SplatWeightMultiplication(rayValue, pxsyi[idx + offset_xs + offset_yi], stepLengthInVoxel, voxelSize, lx * lyc);
  m_SplatWeightMultiplication(rayValue, pxsys[idx + offset_xs + offset_ys], stepLengthInVoxel, voxelSize, lx * ly);
}

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TSplatWeightMultiplication,
          class TSumAlongRay>
typename JosephBackProjectionImageFilter<TInputImage,
                                         TOutputImage,
                                         TInterpolationWeightMultiplication,
                                         TSplatWeightMultiplication,
                                         TSumAlongRay>::OutputPixelType
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TInterpolationWeightMultiplication,
                                TSplatWeightMultiplication,
                                TSumAlongRay>::BilinearInterpolation(const double           stepLengthInVoxel,
                                                                     const InputPixelType * pxiyi,
                                                                     const InputPixelType * pxsyi,
                                                                     const InputPixelType * pxiys,
                                                                     const InputPixelType * pxsys,
                                                                     const CoordRepType     x,
                                                                     const CoordRepType     y,
                                                                     const int              ox,
                                                                     const int              oy)
{
  int          ix = itk::Math::floor(x);
  int          iy = itk::Math::floor(y);
  int          idx = ix * ox + iy * oy;
  CoordRepType lx = x - ix;
  CoordRepType ly = y - iy;
  CoordRepType lxc = 1. - lx;
  CoordRepType lyc = 1. - ly;
  return (m_InterpolationWeightMultiplication(stepLengthInVoxel, lxc * lyc, pxiyi, idx) +
          m_InterpolationWeightMultiplication(stepLengthInVoxel, lx * lyc, pxsyi, idx) +
          m_InterpolationWeightMultiplication(stepLengthInVoxel, lxc * ly, pxiys, idx) +
          m_InterpolationWeightMultiplication(stepLengthInVoxel, lx * ly, pxsys, idx));
}

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TSplatWeightMultiplication,
          class TSumAlongRay>
typename JosephBackProjectionImageFilter<TInputImage,
                                         TOutputImage,
                                         TInterpolationWeightMultiplication,
                                         TSplatWeightMultiplication,
                                         TSumAlongRay>::OutputPixelType
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TInterpolationWeightMultiplication,
                                TSplatWeightMultiplication,
                                TSumAlongRay>::BilinearInterpolationOnBorders(const double           stepLengthInVoxel,
                                                                              const InputPixelType * pxiyi,
                                                                              const InputPixelType * pxsyi,
                                                                              const InputPixelType * pxiys,
                                                                              const InputPixelType * pxsys,
                                                                              const CoordRepType     x,
                                                                              const CoordRepType     y,
                                                                              const int              ox,
                                                                              const int              oy,
                                                                              const CoordRepType     minx,
                                                                              const CoordRepType     miny,
                                                                              const CoordRepType     maxx,
                                                                              const CoordRepType     maxy)
{
  int          ix = itk::Math::floor(x);
  int          iy = itk::Math::floor(y);
  int          idx = ix * ox + iy * oy;
  CoordRepType lx = x - ix;
  CoordRepType ly = y - iy;
  CoordRepType lxc = 1. - lx;
  CoordRepType lyc = 1. - ly;

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

  result += m_InterpolationWeightMultiplication(stepLengthInVoxel, lxc * lyc, pxiyi, idx + offset_xi + offset_yi);
  result += m_InterpolationWeightMultiplication(stepLengthInVoxel, lxc * ly, pxiys, idx + offset_xi + offset_ys);
  result += m_InterpolationWeightMultiplication(stepLengthInVoxel, lx * lyc, pxsyi, idx + offset_xs + offset_yi);
  result += m_InterpolationWeightMultiplication(stepLengthInVoxel, lx * ly, pxsys, idx + offset_xs + offset_ys);

  return (result);
}

} // end namespace rtk

#endif
