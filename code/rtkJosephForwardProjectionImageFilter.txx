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

#ifndef __rtkJosephForwardProjectionImageFilter_txx
#define __rtkJosephForwardProjectionImageFilter_txx

#include "rtkHomogeneousMatrix.h"
#include "rtkRayBoxIntersectionFunction.h"

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
  const unsigned int nPixelPerProj = outputRegionForThread.GetSize(0)*outputRegionForThread.GetSize(1);
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

  // Iterators on input and output projections
  typedef itk::ImageRegionConstIterator<TInputImage> InputRegionIterator;
  InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
  typedef itk::ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Create intersection functions, one for each possible main direction
  typedef rtk::RayBoxIntersectionFunction<CoordRepType, Dimension> RBIFunctionType;
  typename RBIFunctionType::Pointer rbi[Dimension];
  for(unsigned int j=0; j<Dimension; j++)
    {
    rbi[j] = RBIFunctionType::New();
    typename RBIFunctionType::VectorType boxMin, boxMax;
    for(unsigned int i=0; i<Dimension; i++)
      {
      boxMin[i] = this->GetInput(1)->GetBufferedRegion().GetIndex()[i];
      boxMax[i] = this->GetInput(1)->GetBufferedRegion().GetIndex()[i] +
                  this->GetInput(1)->GetBufferedRegion().GetSize()[i] - 1;
      }
    rbi[j]->SetBoxMin(boxMin);
    rbi[j]->SetBoxMax(boxMax);
    }

  // Go over each projection
  for(int iProj=outputRegionForThread.GetIndex(2);
          iProj<outputRegionForThread.GetIndex(2)+(int)outputRegionForThread.GetSize(2);
          iProj++)
    {
    // Account for system rotations
    // volPPToIndex maps the physical 3D coordinates of a point (in mm) to the
    // corresponding 3D volume index
    typename Superclass::GeometryType::ThreeDHomogeneousMatrixType volPPToIndex;
    volPPToIndex = GetPhysicalPointToIndexMatrix( this->GetInput(1) );

    // Set source position in volume indices
    // GetSourcePosition() returns coordinates in mm. Multiplying by 
    // volPPToIndex gives the corresponding volume index
    typename Superclass::GeometryType::HomogeneousVectorType sourcePosition;
    sourcePosition = volPPToIndex * geometry->GetSourcePosition(iProj);
    for(unsigned int i=0; i<Dimension; i++)
      rbi[i]->SetRayOrigin( &(sourcePosition[0]) );

    // Compute matrix to transform projection index to volume index
    // IndexToPhysicalPointMatrix maps the 2D index of a projection's pixel to its 2D position on the detector (in mm)
    // ProjectionCoordinatesToFixedSystemMatrix maps the 2D position of a pixel on the detector to its 3D coordinates in volume's coordinates (still in mm)
    // volPPToIndex maps 3D volume coordinates to a 3D index
    typename Superclass::GeometryType::ThreeDHomogeneousMatrixType matrix;
    matrix = volPPToIndex.GetVnlMatrix() *
             geometry->GetProjectionCoordinatesToFixedSystemMatrix(iProj).GetVnlMatrix() *
             GetIndexToPhysicalPointMatrix( this->GetInput() ).GetVnlMatrix();

    // Go over each pixel of the projection
    typename RBIFunctionType::VectorType dirVox, stepMM, dirVoxAbs, np, fp;
    for(unsigned int pix=0; pix<nPixelPerProj; pix++, ++itIn, ++itOut)
      {
      // Compute point coordinate in volume depending on projection index
      for(unsigned int i=0; i<Dimension; i++)
        {
        dirVox[i] = matrix[i][Dimension];
        for(unsigned int j=0; j<Dimension; j++)
          dirVox[i] += matrix[i][j] * itOut.GetIndex()[j];

        // Direction
        dirVox[i] -= sourcePosition[i];
        }

      // Select main direction
      unsigned int mainDir = 0;
      for(unsigned int i=0; i<Dimension; i++)
        {
        dirVoxAbs[i] = vnl_math_abs( dirVox[i] );
        if(dirVoxAbs[i]>dirVoxAbs[mainDir])
          mainDir = i;
        }

      // Test if there is an intersection
      if( rbi[mainDir]->Evaluate(dirVox) &&
          rbi[mainDir]->GetFarthestDistance()>=0. && // check if detector after the source
          rbi[mainDir]->GetNearestDistance()<=1.)    // check if detector after or in the volume
        {
        // Clip the casting between source and pixel of the detector
        rbi[mainDir]->SetNearestDistance ( std::max(rbi[mainDir]->GetNearestDistance() , 0.) );
        rbi[mainDir]->SetFarthestDistance( std::min(rbi[mainDir]->GetFarthestDistance(), 1.) );

        // Compute and sort intersections: (n)earest and (f)arthest (p)points
        np = rbi[mainDir]->GetNearestPoint();
        fp = rbi[mainDir]->GetFarthestPoint();
        if(np[mainDir]>fp[mainDir])
          std::swap(np, fp);

        // Compute main nearest and farthest slice indices
        // Numerical precision problems occur with the floor function
        // on integer numbers stored as floats.
        // They are avoided by adding a very small epsilon
        float epsilon = 1e-5;
        const int ns = vnl_math_ceil ( np[mainDir] - epsilon);
        const int fs = vnl_math_floor( fp[mainDir] + epsilon);

        // Determine the other two directions
        unsigned int notMainDirInf = (mainDir+1)%Dimension;
        unsigned int notMainDirSup = (mainDir+2)%Dimension;
        if(notMainDirInf>notMainDirSup)
          std::swap(notMainDirInf, notMainDirSup);

        const CoordRepType minx = rbi[mainDir]->GetBoxMin()[notMainDirInf];
        const CoordRepType miny = rbi[mainDir]->GetBoxMin()[notMainDirSup];
        const CoordRepType maxx = rbi[mainDir]->GetBoxMax()[notMainDirInf];
        const CoordRepType maxy = rbi[mainDir]->GetBoxMax()[notMainDirSup];

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
        const CoordRepType residual = ns-np[mainDir];
        const CoordRepType norm = 1/dirVox[mainDir];
        const CoordRepType stepx = dirVox[notMainDirInf] * norm;
        const CoordRepType stepy = dirVox[notMainDirSup] * norm;
        CoordRepType currentx = np[notMainDirInf];
        CoordRepType currenty = np[notMainDirSup];

        // Initialize the accumulation
        typename TOutputImage::PixelType sum = 0;

        // If the ray enters the volume not through the first main direction slice
        if(residual > 0)
          {
          // Move back to the previous main direction slice
          pxiyi -= offsetz;
          pxsyi -= offsetz;
          pxiys -= offsetz;
          pxsys -= offsetz;
          currentx += (residual -1)*stepx;
          currenty += (residual -1)*stepy;

          // Perform bilinear interpolation with padding voxels containing 0
          sum += BilinearInterpolationOnBorders(threadId, 1.,
                                      pxiyi, pxsyi, pxiys, pxsys, currentx, currenty,
                                      offsetx, offsety, minx, miny, maxx, maxy);

          // Move to next main direction slice
          pxiyi += offsetz;
          pxsyi += offsetz;
          pxiys += offsetz;
          pxsys += offsetz;
          currentx += stepx;
          currenty += stepy;

          }

        // Middle steps
        for(int i=ns; i<=fs; i++)
          {

          sum += BilinearInterpolation(threadId,
                                       1.,
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

        // If the ray exits the volume not through the last main direction slice
        if(fp[mainDir] - fs > epsilon)
          {
          sum += BilinearInterpolationOnBorders(threadId,
                                    1.,
                                    pxiyi, pxsyi, pxiys, pxsys,
                                    currentx, currenty, offsetx, offsety, minx, miny, maxx, maxy);
          }

        // Compute voxel to millimeters conversion
        stepMM[notMainDirInf] = this->GetInput(1)->GetSpacing()[notMainDirInf] * stepx;
        stepMM[notMainDirSup] = this->GetInput(1)->GetSpacing()[notMainDirSup] * stepy;
        stepMM[mainDir]       = this->GetInput(1)->GetSpacing()[mainDir];

        // Accumulate
        itOut.Set( m_ProjectedValueAccumulation(threadId,
                                                itIn.Get(),
                                                sum,
                                                stepMM,
                                                &(sourcePosition[0]),
                                                dirVox,
                                                np,
                                                fp) );
        }
      else
        itOut.Set( m_ProjectedValueAccumulation(threadId,
                                                itIn.Get(),
                                                0.,
                                                &(sourcePosition[0]),
                                                &(sourcePosition[0]),
                                                dirVox,
                                                &(sourcePosition[0]),
                                                &(sourcePosition[0])) );
      }
    }
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
  return stepLengthInVoxel * (
           m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * lyc, pxiyi, idx) +
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

  OutputPixelType result=0;
  if(ix >= minx)
    {
    if(iy >= miny)
      result += m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * lyc, pxiyi, idx);
    if(iy+1 <= maxy)
      result += m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lx  * lyc, pxsyi, idx);
    }
  if(ix+1 <= maxx)
    {
    if(iy >= miny)
      result += m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lxc * ly , pxiys, idx);
    if(iy+1 <= maxy)
      result += m_InterpolationWeightMultiplication(threadId, stepLengthInVoxel, lx  * ly , pxsys, idx);
    }
  return (result * stepLengthInVoxel);
}


} // end namespace rtk

#endif
