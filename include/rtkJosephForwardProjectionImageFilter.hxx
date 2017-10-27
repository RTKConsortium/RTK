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

#include "rtkHomogeneousMatrix.h"
#include "rtkRayBoxIntersectionFunction.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>

namespace rtk
{

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TProjectedValueAccumulation>
void
JosephForwardProjectionImageFilter<TInputImage,
                                   TOutputImage,
                                   TInterpolationWeightMultiplication,
                                   TProjectedValueAccumulation>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType threadId )
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  int offsets[3];
  offsets[0] = 1;
  offsets[1] = this->GetInput(1)->GetBufferedRegion().GetSize()[0];
  offsets[2] = this->GetInput(1)->GetBufferedRegion().GetSize()[0] * this->GetInput(1)->GetBufferedRegion().GetSize()[1];
  const typename Superclass::GeometryType::Pointer geometry = this->GetGeometry();

  // beginBuffer is pointing at point with index (0,0,0) in memory, even if
  // it is not in the allocated memory
  const typename TInputImage::PixelType *beginBuffer =
      this->GetInput(1)->GetBufferPointer() -
      offsets[0] * this->GetInput(1)->GetBufferedRegion().GetIndex()[0] -
      offsets[1] * this->GetInput(1)->GetBufferedRegion().GetIndex()[1] -
      offsets[2] * this->GetInput(1)->GetBufferedRegion().GetIndex()[2];

  // volPPToIndex maps the physical 3D coordinates of a point (in mm) to the
  // corresponding 3D volume index
  typename Superclass::GeometryType::ThreeDHomogeneousMatrixType volPPToIndex;
  volPPToIndex = GetPhysicalPointToIndexMatrix( this->GetInput(1) );

  // Iterators on input and output projections
  typedef ProjectionsRegionConstIteratorRayBased<TInputImage> InputRegionIterator;
  InputRegionIterator *itIn;
  itIn = InputRegionIterator::New(this->GetInput(),
                                  outputRegionForThread,
                                  geometry,
                                  volPPToIndex);
  typedef itk::ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Create intersection functions, one for each possible main direction
  typedef rtk::RayBoxIntersectionFunction<CoordRepType, Dimension> RBIFunctionType;
  typename RBIFunctionType::Pointer rbi = RBIFunctionType::New();
  typename RBIFunctionType::VectorType boxMin, boxMax;
  for(unsigned int i=0; i<Dimension; i++)
    {
    boxMin[i] = this->GetInput(1)->GetBufferedRegion().GetIndex()[i];
    boxMax[i] = this->GetInput(1)->GetBufferedRegion().GetIndex()[i] +
                this->GetInput(1)->GetBufferedRegion().GetSize()[i] - 1;
    }
  rbi->SetBoxMin(boxMin);
  rbi->SetBoxMax(boxMax);

  // Go over each pixel of the projection
  typename RBIFunctionType::VectorType stepMM, np, fp;
  for(unsigned int pix=0; pix<outputRegionForThread.GetNumberOfPixels(); pix++, itIn->Next(), ++itOut)
    {
    typename InputRegionIterator::PointType sourcePosition = itIn->GetSourcePosition();
    typename InputRegionIterator::PointType dirVox = itIn->GetSourceToPixel();

    //Set source
    rbi->SetRayOrigin( sourcePosition );

    // Select main direction
    unsigned int mainDir = 0;
    typename RBIFunctionType::VectorType dirVoxAbs;
    for(unsigned int i=0; i<Dimension; i++)
      {
      dirVoxAbs[i] = vnl_math_abs( dirVox[i] );
      if(dirVoxAbs[i]>dirVoxAbs[mainDir])
        mainDir = i;
      }

    // Test if there is an intersection
    if( rbi->Evaluate(&dirVox[0]) &&
        rbi->GetFarthestDistance()>=0. && // check if detector after the source
        rbi->GetNearestDistance()<=1.)    // check if detector after or in the volume
      {
      // Clip the casting between source and pixel of the detector
      rbi->SetNearestDistance ( std::max(rbi->GetNearestDistance() , 0.) );
      rbi->SetFarthestDistance( std::min(rbi->GetFarthestDistance(), 1.) );

      // Compute and sort intersections: (n)earest and (f)arthest (p)points
      np = rbi->GetNearestPoint();
      fp = rbi->GetFarthestPoint();
      if(np[mainDir]>fp[mainDir])
        std::swap(np, fp);

      // Compute main nearest and farthest slice indices
      const int ns = vnl_math_rnd( np[mainDir]);
      const int fs = vnl_math_rnd( fp[mainDir]);

      // Determine the other two directions
      unsigned int notMainDirInf = (mainDir+1)%Dimension;
      unsigned int notMainDirSup = (mainDir+2)%Dimension;
      if(notMainDirInf>notMainDirSup)
        std::swap(notMainDirInf, notMainDirSup);

      const CoordRepType minx = rbi->GetBoxMin()[notMainDirInf];
      const CoordRepType miny = rbi->GetBoxMin()[notMainDirSup];
      const CoordRepType maxx = rbi->GetBoxMax()[notMainDirInf];
      const CoordRepType maxy = rbi->GetBoxMax()[notMainDirSup];

      // Init data pointers to first pixel of slice ns (i)nferior and (s)uperior (x|y) corner
      const int offsetx = offsets[notMainDirInf];
      const int offsety = offsets[notMainDirSup];
      const int offsetz = offsets[mainDir];
      const typename TInputImage::PixelType *pxiyi, *pxsyi, *pxiys, *pxsys;

      pxiyi = beginBuffer + ns * offsetz;
      pxsyi = pxiyi + offsetx;
      pxiys = pxiyi + offsety;
      pxsys = pxsyi + offsety;

      // Compute step size and go to first voxel
      const CoordRepType residual = ns - np[mainDir];
      const CoordRepType norm = 1/dirVox[mainDir];
      const CoordRepType stepx = dirVox[notMainDirInf] * norm;
      const CoordRepType stepy = dirVox[notMainDirSup] * norm;
      CoordRepType currentx = np[notMainDirInf] + residual * stepx;
      CoordRepType currenty = np[notMainDirSup] + residual * stepy;

      // Initialize the accumulation
      typename TOutputImage::PixelType sum = 0;

      if (fs == ns) //If the voxel is a corner, we can skip most steps
        {
          sum += BilinearInterpolationOnBorders(threadId, fp[mainDir] - np[mainDir],
                                                pxiyi, pxsyi, pxiys, pxsys,
                                                currentx, currenty, offsetx, offsety,
                                                minx, miny, maxx, maxy);
        }
      else
        {
        // First step
        sum += BilinearInterpolationOnBorders(threadId, residual + 0.5,
                                              pxiyi, pxsyi, pxiys, pxsys,
                                              currentx, currenty, offsetx, offsety,
                                              minx, miny, maxx, maxy);

        // Move to next main direction slice
        pxiyi += offsetz;
        pxsyi += offsetz;
        pxiys += offsetz;
        pxsys += offsetz;
        currentx += stepx;
        currenty += stepy;

        // Middle steps
        for(int i=ns+1; i<fs; i++)
          {
          sum += BilinearInterpolation(threadId, 1.0,
                                       pxiyi, pxsyi, pxiys, pxsys,
                                       currentx, currenty, offsetx, offsety);

          // Move to next main direction slice
          pxiyi += offsetz;
          pxsyi += offsetz;
          pxiys += offsetz;
          pxsys += offsetz;
          currentx += stepx;
          currenty += stepy;
          }

        // Last step
        sum += BilinearInterpolationOnBorders(threadId,fp[mainDir] - fs + 0.5,
                                              pxiyi, pxsyi, pxiys, pxsys,
                                              currentx, currenty, offsetx, offsety,
                                              minx, miny, maxx, maxy);
        }
      // Compute voxel to millimeters conversion
      stepMM[notMainDirInf] = this->GetInput(1)->GetSpacing()[notMainDirInf] * stepx;
      stepMM[notMainDirSup] = this->GetInput(1)->GetSpacing()[notMainDirSup] * stepy;
      stepMM[mainDir]       = this->GetInput(1)->GetSpacing()[mainDir];

      // Accumulate
      m_ProjectedValueAccumulation(threadId,
                                   itIn->Get(),
                                   itOut.Value(),
                                   sum,
                                   stepMM,
                                   sourcePosition,
                                   dirVox,
                                   np,
                                   fp);
      }
    else
      m_ProjectedValueAccumulation(threadId,
                                   itIn->Get(),
                                   itOut.Value(),
                                   0.,
                                   sourcePosition,
                                   sourcePosition,
                                   dirVox,
                                   sourcePosition,
                                   sourcePosition);
    }
  delete itIn;
}

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TProjectedValueAccumulation>
typename JosephForwardProjectionImageFilter<TInputImage,
                                            TOutputImage,
                                            TInterpolationWeightMultiplication,
                                            TProjectedValueAccumulation>::OutputPixelType
JosephForwardProjectionImageFilter<TInputImage,
                                   TOutputImage,
                                   TInterpolationWeightMultiplication,
                                   TProjectedValueAccumulation>
::BilinearInterpolation( const ThreadIdType threadId,
                         const double stepLengthInVoxel,
                         const InputPixelType *pxiyi,
                         const InputPixelType *pxsyi,
                         const InputPixelType *pxiys,
                         const InputPixelType *pxsys,
                         const CoordRepType x,
                         const CoordRepType y,
                         const int ox,
                         const int oy )
{
  int ix = vnl_math_floor(x);
  int iy = vnl_math_floor(y);
  int idx = ix*ox + iy*oy;
  CoordRepType lx = x - ix;
  CoordRepType ly = y - iy;
  CoordRepType lxc = 1.-lx;
  CoordRepType lyc = 1.-ly;
  return ( m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * lyc, pxiyi, idx) +
           m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lx  * lyc, pxsyi, idx) +
           m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * ly , pxiys, idx) +
           m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lx  * ly , pxsys, idx) );
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
          class TProjectedValueAccumulation>
typename JosephForwardProjectionImageFilter<TInputImage,
                                            TOutputImage,
                                            TInterpolationWeightMultiplication,
                                            TProjectedValueAccumulation>::OutputPixelType
JosephForwardProjectionImageFilter<TInputImage,
                                   TOutputImage,
                                   TInterpolationWeightMultiplication,
                                   TProjectedValueAccumulation>
::BilinearInterpolationOnBorders( const ThreadIdType threadId,
                           const double stepLengthInVoxel,
                           const InputPixelType *pxiyi,
                           const InputPixelType *pxsyi,
                           const InputPixelType *pxiys,
                           const InputPixelType *pxsys,
                           const CoordRepType x,
                           const CoordRepType y,
                           const int ox,
                           const int oy,
                           const CoordRepType minx,
                           const CoordRepType miny,
                           const CoordRepType maxx,
                           const CoordRepType maxy)
{
  int ix = vnl_math_floor(x);
  int iy = vnl_math_floor(y);
  int idx = ix*ox + iy*oy;
  CoordRepType lx = x - ix;
  CoordRepType ly = y - iy;
  CoordRepType lxc = 1.-lx;
  CoordRepType lyc = 1.-ly;

  int offset_xi = 0;
  int offset_yi = 0;
  int offset_xs = 0;
  int offset_ys = 0;

  OutputPixelType result=0;
  if(ix < minx) offset_xi = ox;
  if(iy < miny) offset_yi = oy;
  if(ix >= maxx) offset_xs = -ox;
  if(iy >= maxy) offset_ys = -oy;
  result += m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * lyc, pxiyi, idx + offset_xi + offset_yi);
  result += m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * ly , pxiys, idx + offset_xi + offset_ys);
  result += m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lx  * lyc, pxsyi, idx + offset_xs + offset_yi);
  result += m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lx  * ly , pxsys, idx + offset_xs + offset_ys);

  return (stepLengthInVoxel * result);
}


} // end namespace rtk

#endif
