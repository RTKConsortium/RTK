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

#ifndef rtkJosephForwardProjectionImageFilter_hxx
#define rtkJosephForwardProjectionImageFilter_hxx

#include "rtkJosephForwardProjectionImageFilter.h"

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
  InputRegionIterator * itIn;
  itIn = InputRegionIterator::New(this->GetInput(), outputRegionForThread, geometry, volPPToIndex);
  using OutputRegionIterator = itk::ImageRegionIteratorWithIndex<TOutputImage>;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Create intersection functions, one for each possible main direction
  typename BoxShape::Pointer    box = BoxShape::New();
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

    // Test if there is an intersection
    BoxShape::ScalarType nearDist, farDist;
    if (box->IsIntersectedByRay(pixelPosition, dirVox, nearDist, farDist) &&
        farDist >= 0. && // check if detector after the source
        nearDist <= 1.)  // check if detector after or in the volume
    {
      // Clip the casting between source and pixel of the detector
      nearDist = std::max(nearDist, inferiorClip);
      farDist = std::min(farDist, superiorClip);

      // Compute and sort intersections: (n)earest and (f)arthest (p)points
      np = pixelPosition + nearDist * dirVox;
      fp = pixelPosition + farDist * dirVox;
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
      const int                              offsetx = offsets[notMainDirInf];
      const int                              offsety = offsets[notMainDirSup];
      const int                              offsetz = offsets[mainDir];
      const typename TInputImage::PixelType *pxiyi, *pxsyi, *pxiys, *pxsys;

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
      stepMM[notMainDirInf] = this->GetInput(1)->GetSpacing()[notMainDirInf] * stepx;
      stepMM[notMainDirSup] = this->GetInput(1)->GetSpacing()[notMainDirSup] * stepy;
      stepMM[mainDir] = this->GetInput(1)->GetSpacing()[mainDir];

      // Initialize the accumulation
      typename TOutputImage::PixelType sum = itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue();

      typename TInputImage::PixelType volumeValue = itk::NumericTraits<typename TInputImage::PixelType>::ZeroValue();
      if (fs == ns) // If the voxel is a corner, we can skip most steps
      {
        volumeValue = BilinearInterpolationOnBorders(threadId,
                                                     fp[mainDir] - np[mainDir],
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
        sum += m_SumAlongRay(threadId, volumeValue, stepMM);
      }
      else
      {
        // First step
        volumeValue = BilinearInterpolationOnBorders(threadId,
                                                     residual + 0.5,
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
        sum += m_SumAlongRay(threadId, volumeValue, stepMM);

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
          volumeValue =
            BilinearInterpolation(threadId, 1.0, pxiyi, pxsyi, pxiys, pxsys, currentx, currenty, offsetx, offsety);
          sum += m_SumAlongRay(threadId, volumeValue, stepMM);

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
                                                     fp[mainDir] - fs + 0.5,
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
        sum += m_SumAlongRay(threadId, volumeValue, stepMM);
      }
      // Accumulate
      m_ProjectedValueAccumulation(threadId, itIn->Get(), itOut.Value(), sum, stepMM, pixelPosition, dirVox, np, fp);
    }
    else
      m_ProjectedValueAccumulation(
        threadId, itIn->Get(), itOut.Value(), 0., pixelPosition, pixelPosition, dirVox, pixelPosition, pixelPosition);
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
  return (m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * lyc, pxiyi, idx) +
          m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lx * lyc, pxsyi, idx) +
          m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * ly, pxiys, idx) +
          m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lx * ly, pxsys, idx));
  /* Alternative slower solution
  const unsigned int ix = itk::Math::Floor(x);
  const unsigned int iy = itk::Math::Floor(y);
  const unsigned int idx = ix*ox + iy*oy;
  const CoordRepType a = p1[idx];
  const CoordRepType b = p2[idx] - a;
  const CoordRepType c = p3[idx] - a;
  const CoordRepType lx = x-ix;
  const CoordRepType ly = y-iy;
  const CoordRepType d = p4[idx] - a - b - c;
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
